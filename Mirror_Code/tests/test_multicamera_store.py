"""Strict wire decoding, provenance, durability, and concurrent local ingestion."""

from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sqlite3
import tempfile
from threading import Barrier
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from core.camera_calibration import CameraCalibration
from core.multicamera_fusion import FusedPacket, FusedPerson
from core.multicamera_store import LocalFusionStore, PacketConflictError, decode_fused_packet
from core.multicamera_types import CalibratedCamera, CameraFrame, FrameBundle, PersonObservation


def person(person_id="person", sequence=1, track="track", *, identity=None,
           cameras=("a", "b"), epoch="session"):
    points = np.array([[0.1, .2, 3], [.2, .3, 3.1], [np.nan, np.nan, np.nan]])
    return FusedPerson(
        person_id, identity, "confirmed" if identity is not None else "anonymous",
        points, np.array([True, True, False]),
        (tuple(cameras), tuple(cameras), ()), ("insufficient_views",),
        tuple((camera, epoch, sequence, track) for camera in cameras),
    )


def packet(bundle_id="bundle", sequence=1, track="track", **person_options):
    return FusedPacket(bundle_id, 10.0, (person(sequence=sequence, track=track, **person_options),),
                       ("c",), ("camera_timeout",))


class FusedWireBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.packet = packet()
        self.payload = self.packet.to_dict()

    def assert_invalid(self, payload):
        with self.assertRaises(ValueError):
            decode_fused_packet(json.dumps(payload))

    def test_roundtrip_retains_null_occluded_joints_and_provenance(self):
        wire = self.packet.to_json()
        self.assertNotIn("NaN", wire)
        decoded = decode_fused_packet(wire)
        self.assertEqual(decoded.to_json(), wire)
        self.assertIsNone(json.loads(wire)["persons"][0]["world_points_m"][2])
        self.assertTrue(np.isnan(decoded.persons[0].world_points[2]).all())
        self.assertEqual(decoded.persons[0].observation_keys, (("a", "session", 1, "track"), ("b", "session", 1, "track")))

    def test_duplicate_json_fields_and_nonfinite_constants_are_rejected(self):
        wire = self.packet.to_json()
        invalid = (
            wire.replace('"schema_version":1', '"schema_version":1,"schema_version":1'),
            wire.replace('"person_id":"person"', '"person_id":"person","person_id":"person"'),
            wire.replace('"timestamp_seconds":10.0', '"timestamp_seconds":NaN'),
            wire.replace('"timestamp_seconds":10.0', '"timestamp_seconds":Infinity'),
            wire.replace('"timestamp_seconds":10.0', '"timestamp_seconds":1e400'),
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_fused_packet(value)

    def test_missing_extra_top_level_and_person_fields_are_rejected(self):
        for fields in (self.payload, self.payload["persons"][0]):
            for mode in ("missing", "extra"):
                payload = copy.deepcopy(self.payload)
                target = payload if fields is self.payload else payload["persons"][0]
                if mode == "missing":
                    del target[next(iter(target))]
                else:
                    target["untrusted_extra"] = 1
                self.assert_invalid(payload)
        for payload in ([], None, "object", 1):
            self.assert_invalid(payload)

    def test_schema_and_timestamp_types_are_strict(self):
        for value in (True, "1", 1.0, 0, 2, None):
            payload = copy.deepcopy(self.payload)
            payload["schema_version"] = value
            self.assert_invalid(payload)
        for value in (True, "10", [10], None, -1):
            payload = copy.deepcopy(self.payload)
            payload["timestamp_seconds"] = value
            self.assert_invalid(payload)

    def test_joint_mask_coordinate_and_null_contract_are_enforced(self):
        changes = (
            ("valid_mask", [1, True, False]), ("valid_mask", [True, True]),
            ("valid_mask", "TTF"),
            ("world_points_m", [[0, 0, 3], [0, 0, 3], [0, 0, 0]]),
            ("world_points_m", [None, [0, 0, 3], None]),
            ("world_points_m", [[True, 0, 3], [0, 0, 3], None]),
            ("world_points_m", [["0", 0, 3], [0, 0, 3], None]),
            ("world_points_m", [[0, 3], [0, 0, 3], None]),
        )
        for name, value in changes:
            payload = copy.deepcopy(self.payload)
            payload["persons"][0][name] = value
            self.assert_invalid(payload)

    def test_identifier_support_and_identity_rules_are_enforced(self):
        changes = (
            ("person_id", ""), ("person_id", "x" * 129), ("person_id", "bad\nname"),
            ("identity_status", "guessed"), ("identity_id", "member"),
            ("identity_status", "confirmed"),
            ("joint_support", [["a"], ["a", "b"], []]),
            ("joint_support", [["a", "a"], ["a", "b"], []]),
            ("joint_support", [["a", "unknown"], ["a", "b"], []]),
            ("joint_support", "ab"), ("reasons", "reason"),
        )
        for name, value in changes:
            payload = copy.deepcopy(self.payload)
            payload["persons"][0][name] = value
            self.assert_invalid(payload)

    def test_provenance_fields_types_duplicates_and_bigint_limit(self):
        for keys in (
            [], [["a", "session", 1]], [["a", "session", 1, "track", "extra"]],
            [["a", "session", True, "track"], ["b", "session", 1, "track"]],
            [["a", "session", 1.0, "track"], ["b", "session", 1, "track"]],
            [["a", "session", -1, "track"], ["b", "session", 1, "track"]],
            [["a", "session", 2**63, "track"], ["b", "session", 1, "track"]],
            [["a", "session", 1, "track"], ["a", "session", 1, "track"]],
            [["a", "session", 1, "track"], ["a", "session", 1, "other"]],
        ):
            payload = copy.deepcopy(self.payload)
            payload["persons"][0]["observation_keys"] = keys
            self.assert_invalid(payload)

    def test_packet_identity_person_and_observation_uniqueness(self):
        for alteration in ("person_id", "identity_id", "observation_keys"):
            payload = packet(identity="member").to_dict()
            other = copy.deepcopy(payload["persons"][0])
            other["person_id"] = "other"
            other["identity_id"] = "other-member"
            other["observation_keys"] = [[cam, "session", 1, "other-track"] for cam in ("a", "b")]
            other[alteration] = copy.deepcopy(payload["persons"][0][alteration])
            payload["persons"].append(other)
            self.assert_invalid(payload)

    def test_all_people_from_camera_must_use_same_frame_and_epoch(self):
        for epoch, sequence in (("other-session", 1), ("session", 2)):
            payload = copy.deepcopy(self.payload)
            other = person("other", sequence, "other-track", epoch=epoch).to_dict()
            payload["persons"].append(other)
            self.assert_invalid(payload)

    def test_packet_camera_count_and_metadata_limits_are_enforced(self):
        for name, value in (("persons", [self.payload["persons"][0]] * 65),
                            ("missing_cameras", "c"), ("missing_cameras", ["c"] * 2),
                            ("missing_cameras", ["a"]),
                            ("reasons", "timeout"), ("reasons", ["timeout"] * 2)):
            payload = copy.deepcopy(self.payload)
            payload[name] = value
            self.assert_invalid(payload)
        payload = copy.deepcopy(self.payload)
        payload["missing_cameras"] = []
        payload["persons"] = []
        for index in range(5):
            payload["persons"].append(person(str(index), cameras=(f"a{index}", f"b{index}")).to_dict())
        self.assert_invalid(payload)
        payload = copy.deepcopy(self.payload)
        payload["missing_cameras"] = [f"missing-{index}" for index in range(7)]
        self.assert_invalid(payload)

    def test_joint_count_limit_bytes_not_characters_and_nested_json(self):
        payload = copy.deepcopy(self.payload)
        target = payload["persons"][0]
        target.update(world_points_m=[None] * 257, valid_mask=[False] * 257,
                      joint_support=[[] for _ in range(257)])
        self.assert_invalid(payload)
        payload = copy.deepcopy(self.payload)
        payload["bundle_id"] = "人物"
        wire = json.dumps(payload, ensure_ascii=False)
        with patch("core.multicamera_store.MAX_PACKET_BYTES", len(wire)):
            with self.assertRaises(ValueError):
                decode_fused_packet(wire)
        for wire in (b"{}", "[" * 1500 + "0" + "]" * 1500, "{", "\ud800"):
            with self.assertRaises(ValueError):
                decode_fused_packet(wire)


class LocalFusionStoreTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="multicamera-store-")
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "nested" / "fusion.sqlite3"
        self.store = LocalFusionStore.sqlite(self.path)
        self.addCleanup(self.store.close)

    def test_durable_reopen_duplicate_and_real_null_transport(self):
        original = packet()
        self.assertEqual(self.store.store_json(original.to_json()), "stored")
        self.store.close()
        self.store = LocalFusionStore.sqlite(self.path)
        self.addCleanup(self.store.close)
        self.assertEqual(self.store.store_json(json.dumps(original.to_dict(), indent=2)), "duplicate")
        self.assertEqual(self.store.counts, {"packets": 1, "persons": 1, "observations": 2})
        self.assertEqual(self.store.read_packet("bundle"), original.to_dict())
        self.assertIsNone(self.store.read_packet("absent"))

    def test_conflicting_packet_id_rolls_back_and_preserves_original(self):
        original = packet()
        self.store.store(original)
        replacement = FusedPacket("bundle", 11, original.persons, original.missing_cameras, original.reasons)
        with self.assertRaises(PacketConflictError):
            self.store.store(replacement)
        self.assertEqual(self.store.read_packet("bundle"), original.to_dict())
        self.assertEqual(self.store.counts, {"packets": 1, "persons": 1, "observations": 2})
        self.assertEqual(self.store.store(original), "duplicate")

    def test_observation_conflict_rolls_back_earlier_people_and_provenance(self):
        self.store.store(packet())
        new_person = person("new-person", track="new-track")
        reused_person = person("reused-person")
        mixed = FusedPacket("conflict", 10, (new_person, reused_person))
        with self.assertRaises(PacketConflictError):
            self.store.store(mixed)
        self.assertEqual(self.store.counts, {"packets": 1, "persons": 1, "observations": 2})
        self.assertIsNone(self.store.read_packet("conflict"))
        # A rolled-back transaction must not consume the earlier fresh person.
        self.assertEqual(self.store.store(FusedPacket("fresh", 10, (new_person,))), "stored")
        self.assertEqual(self.store.counts, {"packets": 2, "persons": 2, "observations": 4})

    def test_new_stream_epoch_distinguishes_restarted_sequence_numbers(self):
        self.store.store(packet())
        restarted = FusedPacket("restart", 11, (person(epoch="new-session"),))
        self.assertEqual(self.store.store(restarted), "stored")
        self.assertEqual(self.store.counts["observations"], 4)

    def test_sql_metacharacters_are_data_and_read_results_do_not_alias(self):
        label = "member'); DROP TABLE fusion_packets;--"
        original = packet(bundle_id=label, track=label, identity=label)
        self.assertEqual(self.store.store(original), "stored")
        read = self.store.read_packet(label)
        read["persons"][0]["world_points_m"][0][0] = 999
        self.assertEqual(self.store.read_packet(label), original.to_dict())
        self.assertEqual(self.store.counts["packets"], 1)

    def test_invalid_wire_never_leaves_partial_rows(self):
        payload = packet().to_dict()
        payload["persons"][0]["world_points_m"][2] = [0, 0, 0]
        with self.assertRaises(ValueError):
            self.store.store_json(json.dumps(payload))
        with self.assertRaises(ValueError):
            self.store.store({})
        self.assertEqual(self.store.counts, {"packets": 0, "persons": 0, "observations": 0})

    def test_maximum_bigint_sequence_roundtrips_through_sqlite(self):
        self.assertEqual(self.store.store(packet(sequence=2**63 - 1)), "stored")
        self.assertEqual(self.store.read_packet("bundle")["persons"][0]["observation_keys"][0][2], 2**63 - 1)

    def test_empty_detection_packet_is_durable_without_person_rows(self):
        original = FusedPacket("empty", 1, (), ("a", "b"), ("no_detections",))
        self.assertEqual(self.store.store(original), "stored")
        self.assertEqual(self.store.counts, {"packets": 1, "persons": 0, "observations": 0})

    def test_same_store_concurrent_duplicates_are_idempotent(self):
        original = packet()
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda _: self.store.store(original), range(40)))
        self.assertEqual(results.count("stored"), 1)
        self.assertEqual(results.count("duplicate"), 39)
        self.assertEqual(self.store.counts, {"packets": 1, "persons": 1, "observations": 2})

    def test_two_connections_concurrently_store_distinct_packets(self):
        second = LocalFusionStore.sqlite(self.path)
        self.addCleanup(second.close)
        barrier = Barrier(2)

        def write_batch(store, offset):
            barrier.wait(timeout=5)
            return [store.store(packet(f"packet-{index}", sequence=index))
                    for index in range(offset, offset + 20)]

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(write_batch, self.store, 1)
            other = executor.submit(write_batch, second, 21)
            self.assertEqual(first.result(timeout=15), ["stored"] * 20)
            self.assertEqual(other.result(timeout=15), ["stored"] * 20)
        self.assertEqual(self.store.counts, {"packets": 40, "persons": 40, "observations": 80})

    def test_two_connection_race_rejects_reused_observation_atomically(self):
        second = LocalFusionStore.sqlite(self.path)
        self.addCleanup(second.close)
        barrier = Barrier(2)

        def attempt(store, bundle_id):
            barrier.wait(timeout=5)
            try:
                return store.store(packet(bundle_id))
            except PacketConflictError:
                return "conflict"

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(attempt, self.store, "one")
            other = executor.submit(attempt, second, "two")
            results = [first.result(timeout=15), other.result(timeout=15)]
        self.assertCountEqual(results, ["stored", "conflict"])
        self.assertEqual(self.store.counts, {"packets": 1, "persons": 1, "observations": 2})

    def test_two_connection_duplicate_race_commits_exactly_once(self):
        second = LocalFusionStore.sqlite(self.path)
        self.addCleanup(second.close)
        barrier = Barrier(2)

        def attempt(store):
            barrier.wait(timeout=5)
            return store.store(packet())

        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(attempt, self.store)
            other = executor.submit(attempt, second)
            self.assertCountEqual([first.result(timeout=15), other.result(timeout=15)], ["stored", "duplicate"])
        self.assertEqual(self.store.counts, {"packets": 1, "persons": 1, "observations": 2})

    def test_close_is_idempotent_and_rejects_use_after_close(self):
        self.store.close()
        self.store.close()
        for action in (lambda: self.store.counts, lambda: self.store.read_packet("bundle"),
                       lambda: self.store.store(packet())):
            with self.assertRaises(RuntimeError):
                action()

    def test_external_connection_is_retained_when_store_closes(self):
        connection = sqlite3.connect(":memory:")
        self.addCleanup(connection.close)
        store = LocalFusionStore(connection, "sqlite")
        store.close()
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM fusion_packets").fetchone()[0], 0)

    def test_database_failure_after_packet_insert_rolls_back_every_table(self):
        # A real SQLite trigger fails the second table insert, exercising the
        # actual SQL transaction rather than a mocked store implementation.
        self.store.connection.execute("""CREATE TRIGGER force_person_failure BEFORE INSERT ON fusion_people
            BEGIN SELECT RAISE(ABORT, 'injected person failure'); END""")
        self.store.connection.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.store(packet())
        self.assertEqual(self.store.counts, {"packets": 0, "persons": 0, "observations": 0})
        self.store.connection.execute("DROP TRIGGER force_person_failure")
        self.store.connection.commit()
        self.assertEqual(self.store.store(packet()), "stored")

    def test_sqlite_schema_failure_closes_owned_connection_and_cursor(self):
        connection, cursor = Mock(), Mock()
        connection.cursor.return_value = cursor
        cursor.execute.side_effect = [None, None, sqlite3.OperationalError("injected schema failure")]
        with patch("core.multicamera_store.sqlite3.connect", return_value=connection):
            with self.assertRaises(sqlite3.OperationalError):
                LocalFusionStore.sqlite(":memory:")
        connection.rollback.assert_called_once()
        connection.close.assert_called_once()
        cursor.close.assert_called_once()

    def test_sqlite_pragma_failure_closes_owned_connection(self):
        connection = Mock()
        connection.execute.side_effect = sqlite3.OperationalError("injected pragma failure")
        with patch("core.multicamera_store.sqlite3.connect", return_value=connection):
            with self.assertRaises(sqlite3.OperationalError):
                LocalFusionStore.sqlite(":memory:")
        connection.close.assert_called_once()

    def test_postgres_schema_failure_closes_owned_connection(self):
        connection, cursor = Mock(), Mock()
        connection.cursor.return_value = cursor
        cursor.execute.side_effect = RuntimeError("injected schema failure")
        driver = SimpleNamespace(connect=Mock(return_value=connection))
        with patch.dict("sys.modules", {"psycopg2": driver}):
            with self.assertRaises(RuntimeError):
                LocalFusionStore.postgres("dbname=fake_test")
        driver.connect.assert_called_once_with("dbname=fake_test", connect_timeout=8)
        connection.rollback.assert_called_once()
        connection.close.assert_called_once()
        cursor.close.assert_called_once()

    def test_failed_commit_rolls_back_rows_and_connection_can_recover(self):
        class FailingCommitConnection(sqlite3.Connection):
            fail_commit = False

            def commit(self):
                if self.fail_commit:
                    raise sqlite3.OperationalError("injected commit failure")
                return super().commit()

        connection = sqlite3.connect(":memory:", factory=FailingCommitConnection)
        self.addCleanup(connection.close)
        store = LocalFusionStore(connection, "sqlite")
        self.addCleanup(store.close)
        connection.fail_commit = True
        with self.assertRaises(sqlite3.OperationalError):
            store.store(packet())
        connection.fail_commit = False
        self.assertEqual(store.counts, {"packets": 0, "persons": 0, "observations": 0})
        self.assertEqual(store.store(packet()), "stored")


class MultiCameraPacketTypeTests(unittest.TestCase):
    def setUp(self):
        self.calibration = CameraCalibration(np.array([[800., 0, 640], [0, 800, 480], [0, 0, 1]]),
                                             np.zeros(5), (1280, 960))

    def test_camera_and_observation_arrays_do_not_alias_callers(self):
        rotation, translation = np.eye(3), np.zeros(3)
        camera = CalibratedCamera("camera", self.calibration, rotation, translation)
        points, confidence = np.ones((3, 2)), np.ones(3)
        observation = PersonObservation("track", points, confidence)
        rotation[0, 0], translation[0], points[0, 0], confidence[0] = 99, 99, 99, 0
        self.assertEqual(camera.rotation[0, 0], 1)
        self.assertEqual(camera.translation_m[0], 0)
        self.assertEqual(observation.landmarks_px[0, 0], 1)
        self.assertEqual(observation.confidence[0], 1)
        for array in (camera.rotation, camera.translation_m, observation.landmarks_px, observation.confidence):
            with self.assertRaises(ValueError):
                array.flat[0] = 9

    def test_fused_arrays_and_nested_provenance_do_not_alias_callers(self):
        points = np.array([[0., 0, 3], [0, 1, 3], [0, 2, 3]])
        mask = np.array([True, True, False])
        support = [["a", "b"], ["a", "b"], []]
        keys = [["a", "epoch", 1, "track"], ["b", "epoch", 1, "track"]]
        value = FusedPerson("person", None, "anonymous", points, mask, support, [], keys)
        points[0, 0], mask[0], keys[0][0] = 100, False, "changed"
        support[0].clear()
        self.assertEqual(value.world_points[0, 0], 0)
        self.assertTrue(value.valid_mask[0])
        self.assertEqual(value.joint_support[0], ("a", "b"))
        self.assertEqual(value.observation_keys[0][0], "a")
        self.assertTrue(np.isnan(value.world_points[2]).all())
        for array in (value.world_points, value.valid_mask):
            with self.assertRaises(ValueError):
                array.flat[0] = 1

    def test_missing_observations_and_identity_evidence_are_validated(self):
        for points, confidence, identity, identity_confidence in (
            ([[np.nan, 1], [1, 1], [1, 1]], [1, 1, 1], None, 0),
            ([[np.inf, 1], [1, 1], [1, 1]], [0, 1, 1], None, 0),
            (np.ones((3, 2)), [1, -1, 1], None, 0),
            (np.ones((3, 2)), [1, 1, 1], None, .9),
            (np.ones((3, 2)), [1, 1, 1], "member", 1.1),
        ):
            with self.assertRaises(ValueError):
                PersonObservation("track", points, confidence, identity, identity_confidence)
        value = PersonObservation("track", [[np.nan, np.nan], [1, 1], [1, 1]], [0, 1, 1])
        self.assertEqual(value.confidence[0], 0)

    def test_frame_sequence_domain_and_duplicate_local_tracks(self):
        observation = PersonObservation("track", np.ones((3, 2)), np.ones(3))
        for sequence in (True, -1, 1.2, 2**63):
            with self.assertRaises(ValueError):
                CameraFrame("camera", sequence, 1, 1, (observation,))
        for observations in ((observation, observation), (observation,) * 65):
            with self.assertRaises(ValueError):
                CameraFrame("camera", 1, 1, 1, observations)
        self.assertEqual(CameraFrame("camera", 2**63 - 1, 1, 1, ()).sequence, 2**63 - 1)

    def test_frame_and_bundle_container_inputs_are_copied(self):
        observation = PersonObservation("track", np.ones((3, 2)), np.ones(3))
        observations = [observation]
        frame = CameraFrame("camera", 1, 1, 1, observations)
        observations.clear()
        frames, missing = [frame], ["absent"]
        bundle = FrameBundle("bundle", 1, frames, missing, 0)
        frames.clear()
        missing.clear()
        self.assertEqual(len(frame.observations), 1)
        self.assertEqual(bundle.frames, (frame,))
        self.assertEqual(bundle.missing_cameras, ("absent",))
        with self.assertRaises(ValueError):
            FrameBundle("invalid", 1, (frame, frame), (), 0)
        with self.assertRaises(ValueError):
            FrameBundle("invalid", 1, (frame,), ("camera",), 0)


if __name__ == "__main__":
    unittest.main()
