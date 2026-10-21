"""Durable, idempotent local ingestion of validated fused pose wire packets.

This is an explicit offline/local fusion sink, separate from exercise_data. It
does not send anonymous or ambiguous observations into member exercise history.
PostgreSQL uses a dedicated connection; SQLite makes recording replay repeatable
without external services. Both backends use the same transaction/provenance rules.
"""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import os
import sqlite3
from threading import RLock

import numpy as np

from .multicamera_fusion import FusedPacket, FusedPerson

MAX_PACKET_BYTES = 8 * 1024 * 1024


class PacketConflictError(ValueError):
    """A packet ID or source observation has already been used differently."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON value: {value}")


def decode_fused_packet(wire_payload):
    """Validate the actual JSON transport boundary, including null joint data."""
    if not isinstance(wire_payload, str) or len(wire_payload.encode("utf-8")) > MAX_PACKET_BYTES:
        raise ValueError("fused packet must be UTF-8 JSON within the 8 MiB limit")
    try:
        payload = json.loads(wire_payload, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid fused packet JSON") from exc
    expected = {"schema_version", "bundle_id", "timestamp_seconds", "persons", "missing_cameras", "reasons"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("unexpected or missing fused packet fields")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("unsupported fused packet schema")
    if isinstance(payload["timestamp_seconds"], bool) or not isinstance(payload["timestamp_seconds"], (int, float)):
        raise ValueError("timestamp_seconds must be a JSON number")
    if not isinstance(payload["persons"], list) or len(payload["persons"]) > 64:
        raise ValueError("invalid fused person list")
    persons = []
    person_fields = {"person_id", "identity_id", "identity_status", "world_points_m", "valid_mask", "joint_support", "reasons", "observation_keys"}
    for person in payload["persons"]:
        if not isinstance(person, dict) or set(person) != person_fields:
            raise ValueError("unexpected or missing fused person fields")
        raw_points = person["world_points_m"]
        valid = person["valid_mask"]
        if (not isinstance(raw_points, list) or not isinstance(valid, list)
                or len(raw_points) != len(valid) or not 3 <= len(valid) <= 256
                or any(type(value) is not bool for value in valid)):
            raise ValueError("invalid joint coordinates or mask")
        points = []
        for point, available in zip(raw_points, valid):
            if available:
                if (not isinstance(point, list) or len(point) != 3
                        or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in point)):
                    raise ValueError("a valid joint requires three numeric metric coordinates")
                points.append(point)
            else:
                if point is not None:
                    raise ValueError("unavailable metric joints must be JSON null")
                points.append([np.nan, np.nan, np.nan])
        for name in ("joint_support", "reasons", "observation_keys"):
            if not isinstance(person[name], list):
                raise ValueError(f"{name} must be a JSON list")
        persons.append(FusedPerson(
            person["person_id"], person["identity_id"], person["identity_status"],
            points, valid, person["joint_support"], person["reasons"], person["observation_keys"],
        ))
    if not isinstance(payload["missing_cameras"], list) or not isinstance(payload["reasons"], list):
        raise ValueError("packet metadata must be lists")
    return FusedPacket(payload["bundle_id"], payload["timestamp_seconds"], tuple(persons),
                       tuple(payload["missing_cameras"]), tuple(payload["reasons"]))


class LocalFusionStore:
    """Atomically store each fused packet and each source observation once."""

    def __init__(self, connection, backend, *, owns_connection=False):
        if backend not in {"sqlite", "postgres"}:
            raise ValueError("unsupported fusion storage backend")
        self.connection = connection
        self.backend = backend
        self._owns_connection = owns_connection
        self._lock = RLock()
        self._placeholder = "?" if backend == "sqlite" else "%s"
        self._closed = False
        self._ensure_schema()

    @classmethod
    def sqlite(cls, path):
        if str(path) != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(str(path), check_same_thread=False, timeout=10)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA journal_mode = WAL")
            if str(path) != ":memory:" and os.name != "nt":
                os.chmod(path, 0o600)
            return cls(connection, "sqlite", owns_connection=True)
        except BaseException:
            connection.close()
            raise

    @classmethod
    def postgres(cls, dsn):
        import psycopg2
        connection = psycopg2.connect(dsn, connect_timeout=8)
        try:
            return cls(connection, "postgres", owns_connection=True)
        except BaseException:
            connection.close()
            raise

    @contextmanager
    def _transaction(self):
        with self._lock:
            if self._closed:
                raise RuntimeError("fusion store is closed")
            cursor = self.connection.cursor()
            try:
                yield cursor
                self.connection.commit()
            except BaseException:
                self.connection.rollback()
                raise
            finally:
                cursor.close()

    def _ensure_schema(self):
        with self._transaction() as cursor:
            cursor.execute("""CREATE TABLE IF NOT EXISTS fusion_packets (
                packet_id TEXT PRIMARY KEY, capture_time DOUBLE PRECISION NOT NULL,
                payload_hash TEXT NOT NULL, payload_json TEXT NOT NULL)""")
            cursor.execute("""CREATE TABLE IF NOT EXISTS fusion_people (
                packet_id TEXT NOT NULL REFERENCES fusion_packets(packet_id) ON DELETE CASCADE,
                person_id TEXT NOT NULL, identity_id TEXT, identity_status TEXT NOT NULL,
                valid_joint_count INTEGER NOT NULL,
                PRIMARY KEY (packet_id, person_id))""")
            cursor.execute("""CREATE TABLE IF NOT EXISTS fusion_observations (
                camera_id TEXT NOT NULL, stream_epoch TEXT NOT NULL, sequence BIGINT NOT NULL,
                local_track_id TEXT NOT NULL, packet_id TEXT NOT NULL, person_id TEXT NOT NULL,
                PRIMARY KEY (camera_id, stream_epoch, sequence, local_track_id),
                FOREIGN KEY (packet_id, person_id) REFERENCES fusion_people(packet_id, person_id)
                    ON DELETE CASCADE)""")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_fusion_identity ON fusion_people(identity_id)")

    def store_json(self, wire_payload):
        return self.store(decode_fused_packet(wire_payload))

    def store(self, packet):
        if not isinstance(packet, FusedPacket):
            raise ValueError("store requires a validated FusedPacket")
        # Round-trip through the same strict wire validator for object callers.
        canonical = decode_fused_packet(packet.to_json()).to_json()
        if len(canonical.encode("utf-8")) > MAX_PACKET_BYTES:
            raise ValueError("fused packet exceeds the 8 MiB limit")
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        p = self._placeholder
        try:
            with self._transaction() as cursor:
                cursor.execute(
                    f"INSERT INTO fusion_packets (packet_id,capture_time,payload_hash,payload_json) VALUES ({p},{p},{p},{p}) ON CONFLICT (packet_id) DO NOTHING",
                    (packet.bundle_id, packet.timestamp_seconds, digest, canonical),
                )
                if cursor.rowcount == 0:
                    cursor.execute(f"SELECT payload_hash,payload_json FROM fusion_packets WHERE packet_id={p}", (packet.bundle_id,))
                    stored = cursor.fetchone()
                    if stored == (digest, canonical):
                        return "duplicate"
                    raise PacketConflictError("packet ID already belongs to different fused observations")
                observation_keys = set()
                for person in packet.persons:
                    cursor.execute(
                        f"INSERT INTO fusion_people (packet_id,person_id,identity_id,identity_status,valid_joint_count) VALUES ({p},{p},{p},{p},{p})",
                        (packet.bundle_id, person.person_id, person.identity_id, person.identity_status, int(np.count_nonzero(person.valid_mask))),
                    )
                    for key in person.observation_keys:
                        if tuple(key) in observation_keys:
                            raise PacketConflictError("one camera observation cannot belong to two people")
                        observation_keys.add(tuple(key))
                        cursor.execute(
                            f"INSERT INTO fusion_observations (camera_id,stream_epoch,sequence,local_track_id,packet_id,person_id) VALUES ({p},{p},{p},{p},{p},{p}) ON CONFLICT (camera_id,stream_epoch,sequence,local_track_id) DO NOTHING",
                            (*key, packet.bundle_id, person.person_id),
                        )
                        if cursor.rowcount == 0:
                            raise PacketConflictError("a source observation was already ingested in another packet")
            return "stored"
        except PacketConflictError:
            raise

    @property
    def counts(self):
        with self._transaction() as cursor:
            result = {}
            for label, table in (("packets", "fusion_packets"), ("persons", "fusion_people"), ("observations", "fusion_observations")):
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                result[label] = int(cursor.fetchone()[0])
            return result

    def read_packet(self, packet_id):
        with self._transaction() as cursor:
            cursor.execute(f"SELECT payload_json FROM fusion_packets WHERE packet_id={self._placeholder}", (packet_id,))
            row = cursor.fetchone()
            return None if row is None else json.loads(row[0])

    def close(self):
        with self._lock:
            if not self._closed and self._owns_connection:
                self.connection.close()
            self._closed = True
