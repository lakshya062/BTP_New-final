#!/usr/bin/env python3
"""Verify fusion ingestion against an explicitly supplied disposable PostgreSQL DB.

Set SMART_GYM_FUSION_TEST_PG_DSN to the test database connection string. There is
deliberately no fallback to the application's configured database. The verifier
creates the fusion tables if absent and retains UUID-namespaced evidence rows;
it never deletes or truncates data. It tests storage, not live camera inference.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
from threading import Barrier
from uuid import uuid4

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / "Mirror_Code"))

from core.multicamera_fusion import FusedPacket, FusedPerson  # noqa: E402
from core.multicamera_store import LocalFusionStore, PacketConflictError  # noqa: E402


class VerificationFailure(AssertionError):
    """A required storage invariant did not hold."""


def require(condition, message):
    if not condition:
        raise VerificationFailure(message)


def make_packet(namespace, label, *, source_label=None):
    """Three-joint fixture with one unavailable joint and two visible views."""
    camera_a, camera_b = f"{namespace}-a", f"{namespace}-b"
    epoch = f"{namespace}-{source_label if source_label is not None else label}"
    keys = ((camera_a, epoch, 1, "local-a"), (camera_b, epoch, 1, "local-b"))
    person = FusedPerson(
        f"{namespace}-{label}-person", f"{namespace}-member", "confirmed",
        np.array([[0.0, 0.0, 4.0], [np.nan, np.nan, np.nan], [0.2, 0.3, 4.1]]),
        np.array([True, False, True]), ((camera_a, camera_b), (), (camera_a, camera_b)),
        ("insufficient_views",), keys,
    )
    return FusedPacket(f"{namespace}-{label}", 10.0, (person,), (), ())


def namespace_counts(store, namespace):
    """Exclude unrelated test runs and existing application data from assertions."""
    counts = {}
    with store._transaction() as cursor:
        for name, table in (("packets", "fusion_packets"), ("persons", "fusion_people"),
                            ("observations", "fusion_observations")):
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE packet_id LIKE %s", (namespace + "-%",))
            counts[name] = int(cursor.fetchone()[0])
    return counts


def require_conflict(store, packet):
    try:
        store.store(packet)
    except PacketConflictError:
        return
    raise VerificationFailure("conflicting packet was accepted")


def concurrent_ingest(stores, packets):
    """Require two independent sessions to start the ingestion race together."""
    require(len(stores) == len(packets) == 2, "a concurrency check needs two sessions")
    require(stores[0].connection is not stores[1].connection, "connections must be independent")
    barrier = Barrier(2)

    def ingest(store, packet):
        barrier.wait(timeout=10)
        try:
            return store.store(packet)
        except PacketConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(ingest, store, packet) for store, packet in zip(stores, packets)]
        return [future.result(timeout=30) for future in futures]


def run_verification(dsn):
    import psycopg2

    namespace = "fusion-verify-" + uuid4().hex
    report = {
        "schema_version": 1, "backend": "postgres", "namespace": namespace,
        "evidence_rows_retained": True, "checks": [], "passed": False,
    }
    stores = []
    primary = None

    def connect():
        store = LocalFusionStore.postgres(dsn)
        stores.append(store)
        with store._transaction() as cursor:
            cursor.execute("SET statement_timeout = '15s'")
            cursor.execute("SET lock_timeout = '10s'")
            cursor.execute("SET idle_in_transaction_session_timeout = '15s'")
        return store

    def check(name, operation):
        try:
            details = operation()
            report["checks"].append({"name": name, "passed": True, "details": details})
            return True
        except Exception as exc:
            # Do not echo DSNs, authentication errors, or database endpoints.
            result = {"name": name, "passed": False, "error_type": type(exc).__name__}
            if isinstance(exc, VerificationFailure):
                result["message"] = str(exc)
            report["checks"].append(result)
            return False

    try:
        primary = connect()
        base = make_packet(namespace, "roundtrip")

        def roundtrip():
            require(namespace_counts(primary, namespace) == {"packets": 0, "persons": 0, "observations": 0},
                    "UUID namespace was not empty before the test")
            require(primary.store_json(base.to_json()) == "stored", "initial JSON packet was not stored")
            saved = primary.read_packet(base.bundle_id)
            require(saved == base.to_dict(), "JSON round-trip changed packet contents")
            points = saved["persons"][0]["world_points_m"]
            require(len(points) == 3 and points[1] is None and points[0] is not None and points[2] is not None,
                    "missing-joint null or valid coordinates were not preserved")
            require("NaN" not in json.dumps(saved, allow_nan=False), "nonfinite JSON escaped the wire boundary")
            require(namespace_counts(primary, namespace) == {"packets": 1, "persons": 1, "observations": 2},
                    "initial packet did not persist exactly one person and two source observations")
            return {"packet_id": base.bundle_id, "valid_joints": 2, "null_joints": 1}

        require(check("json_roundtrip_and_null_geometry", roundtrip), "initial persistence check failed")

        def reopen_duplicate():
            nonlocal primary
            primary.close()
            require(primary.connection.closed != 0, "owned PostgreSQL connection was not closed")
            primary = connect()
            require(primary.store(base) == "duplicate", "durable replay after reopen was not a duplicate")
            require(primary.read_packet(base.bundle_id) == base.to_dict(), "durable packet changed after reopen")
            require(namespace_counts(primary, namespace) == {"packets": 1, "persons": 1, "observations": 2},
                    "duplicate replay changed persisted row counts")
            return {"result": "duplicate", "reopened": True}

        check("durable_reopen_and_duplicate", reopen_duplicate)

        def packet_id_conflict():
            before = namespace_counts(primary, namespace)
            require_conflict(primary, replace(base, timestamp_seconds=11.0))
            require(primary.read_packet(base.bundle_id) == base.to_dict(), "ID conflict overwrote the original packet")
            require(namespace_counts(primary, namespace) == before, "ID conflict leaked partial rows")
            return {"original_preserved": True, "row_counts_unchanged": True}

        check("conflicting_packet_id_rolls_back", packet_id_conflict)

        def source_reuse():
            before = namespace_counts(primary, namespace)
            rejected = make_packet(namespace, "source-reuse", source_label="roundtrip")
            require_conflict(primary, rejected)
            require(primary.read_packet(rejected.bundle_id) is None, "source-reuse transaction left its packet row")
            require(namespace_counts(primary, namespace) == before, "source-reuse rollback leaked person or source rows")
            return {"rejected_packet_id": rejected.bundle_id, "row_counts_unchanged": True}

        check("source_observation_reuse_rolls_back", source_reuse)

        def transaction_error_recovery():
            before = namespace_counts(primary, namespace)
            marker = make_packet(namespace, "rollback-marker")
            canonical = marker.to_json()
            caught = False
            try:
                with primary._transaction() as cursor:
                    cursor.execute(
                        "INSERT INTO fusion_packets (packet_id,capture_time,payload_hash,payload_json) VALUES (%s,%s,%s,%s)",
                        (marker.bundle_id, marker.timestamp_seconds, hashlib.sha256(canonical.encode()).hexdigest(), canonical),
                    )
                    cursor.execute("SELECT 1 / 0")  # Safe PostgreSQL error; forces transaction abort.
            except psycopg2.errors.DivisionByZero:
                caught = True
            require(caught, "the real PostgreSQL transaction error was not observed")
            require(primary.read_packet(marker.bundle_id) is None, "database error did not roll back the inserted marker")
            require(namespace_counts(primary, namespace) == before, "database-error rollback changed stored evidence")
            recovery = make_packet(namespace, "recovered")
            require(primary.store(recovery) == "stored", "connection did not recover after database rollback")
            require(primary.read_packet(recovery.bundle_id) == recovery.to_dict(), "recovery packet did not persist")
            require(namespace_counts(primary, namespace) == {
                "packets": before["packets"] + 1, "persons": before["persons"] + 1,
                "observations": before["observations"] + 2,
            }, "recovery transaction produced unexpected row counts")
            return {"database_error": "DivisionByZero", "recovery_packet_id": recovery.bundle_id}

        check("postgres_error_rollback_and_connection_recovery", transaction_error_recovery)

        # Open both connections before racing. Their schema setup is intentionally
        # sequential; this check concerns packet transaction contention.
        competitor = connect()

        def duplicate_race():
            before = namespace_counts(primary, namespace)
            packet = make_packet(namespace, "concurrent-duplicate")
            statuses = concurrent_ingest((primary, competitor), (packet, packet))
            require(sorted(statuses) == ["duplicate", "stored"], "concurrent replay did not store exactly once")
            require(primary.read_packet(packet.bundle_id) == packet.to_dict(), "concurrent replay changed stored packet")
            require(namespace_counts(primary, namespace) == {
                "packets": before["packets"] + 1, "persons": before["persons"] + 1,
                "observations": before["observations"] + 2,
            }, "concurrent replay created duplicate or partial rows")
            return {"statuses": statuses, "packet_id": packet.bundle_id}

        check("two_connection_concurrent_duplicate", duplicate_race)

        def provenance_race():
            before = namespace_counts(primary, namespace)
            packets = (make_packet(namespace, "race-first", source_label="race-shared"),
                       make_packet(namespace, "race-second", source_label="race-shared"))
            statuses = concurrent_ingest((primary, competitor), packets)
            require(sorted(statuses) == ["conflict", "stored"], "contested source observations did not have exactly one winner")
            winner = packets[statuses.index("stored")]
            loser = packets[statuses.index("conflict")]
            require(primary.read_packet(winner.bundle_id) == winner.to_dict(), "winning source transaction did not persist")
            require(primary.read_packet(loser.bundle_id) is None, "losing source transaction leaked its packet row")
            require(namespace_counts(primary, namespace) == {
                "packets": before["packets"] + 1, "persons": before["persons"] + 1,
                "observations": before["observations"] + 2,
            }, "contested-source transaction left duplicate or orphan rows")
            return {"statuses": statuses, "winner_packet_id": winner.bundle_id, "rejected_packet_id": loser.bundle_id}

        check("two_connection_conflicting_provenance", provenance_race)
        report["namespace_row_counts"] = namespace_counts(primary, namespace)
        report["passed"] = (len(report["checks"]) == 7 and all(item["passed"] for item in report["checks"])
                            and report["namespace_row_counts"] == {"packets": 4, "persons": 4, "observations": 8})
    except Exception as exc:
        report["fatal_error_type"] = type(exc).__name__
        if isinstance(exc, VerificationFailure):
            report["fatal_message"] = str(exc)
    finally:
        closed = True
        for store in stores:
            try:
                store.close()
                closed = closed and bool(store.connection.closed)
            except Exception:
                closed = False
        report["owned_connections_closed"] = closed and bool(stores)
        report["passed"] = report["passed"] and report["owned_connections_closed"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="Optional folder for the UUID-namespaced JSON evidence report")
    args = parser.parse_args(argv)
    dsn = os.environ.get("SMART_GYM_FUSION_TEST_PG_DSN")
    if not dsn or not dsn.strip():
        parser.error("SMART_GYM_FUSION_TEST_PG_DSN is required; supply an explicit disposable PostgreSQL test database")
    try:
        report = run_verification(dsn)
    except ImportError as exc:
        report = {"schema_version": 1, "backend": "postgres", "passed": False,
                  "fatal_error_type": type(exc).__name__, "message": "Required PostgreSQL or numerical dependencies are unavailable"}
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.output_dir / f"{report.get('namespace', 'fusion-verification-failed')}.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
