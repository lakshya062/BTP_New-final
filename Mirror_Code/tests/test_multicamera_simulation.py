"""Real synchronization -> association -> geometry -> JSON -> durable SQL replay."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core.multicamera_store import LocalFusionStore, decode_fused_packet


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "simulate_multicamera.py"
SPEC = importlib.util.spec_from_file_location("simulate_multicamera_e2e", SCRIPT)
simulation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(simulation)


class MultiCameraSimulationTests(unittest.TestCase):
    FRAMES = 5  # An odd count includes identical actor positions at the crossing.

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="multicamera-e2e-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.output = Path(cls.directory.name)
        cls.report = simulation.run_simulation(cls.output, seeds=(42,), frames=cls.FRAMES,
                                               calibrated=False, plots=False)
        cls.rows = {row["scenario"]: row for row in cls.report["results"]}
        cls.database = Path(cls.report["local_storage"]["file"])
        with sqlite3.connect(f"file:{cls.database}?mode=ro", uri=True) as connection:
            cls.payloads = [json.loads(row[0]) for row in connection.execute(
                "SELECT payload_json FROM fusion_packets ORDER BY rowid")]
        cls.by_scenario = {}
        for scenario, row in cls.rows.items():
            session = row["sample_packet"]["bundle_id"].split(":")[0]
            cls.by_scenario[scenario] = [payload for payload in cls.payloads
                                       if payload["bundle_id"].startswith(session + ":")]

    def test_all_scenarios_exercise_real_pipeline_and_persist_matching_report(self):
        report = self.report
        self.assertTrue(report["passed"], [(row["scenario"], row["checks"]) for row in report["results"]])
        self.assertEqual(set(self.rows), set(simulation.SCENARIOS))
        self.assertEqual(report["scenario_runs"], len(simulation.SCENARIOS))
        self.assertEqual(report["seeds"], [42])
        self.assertEqual(report["totals"]["bundles"], self.FRAMES * len(simulation.SCENARIOS))
        self.assertEqual(json.loads((self.output / "report.json").read_text()), report)
        self.assertTrue(self.database.is_file())
        for row in report["results"]:
            with self.subTest(scenario=row["scenario"]):
                self.assertEqual(row["bundles"], self.FRAMES)
                self.assertEqual(row["stored_packets"], self.FRAMES)
                self.assertEqual(row["duplicate_packet_deliveries"], self.FRAMES)
                self.assertEqual(len(self.by_scenario[row["scenario"]]), self.FRAMES)
                self.assertTrue(all(row["checks"].values()), row["checks"])
                self.assertTrue(row["checks"]["all_source_observations_accounted_for"])
                self.assertLessEqual(row["max_capture_span_ms"], 12 + 1e-9)
                self.assertEqual(row["mixed_person_groups"], 0)
                self.assertEqual(row["wrong_identity_assignments"], 0)
                self.assertEqual(row["sync_stats"]["accepted_frames"], row["sync_stats"]["emitted_frames"])

    def test_expected_geometry_and_angles_are_nonvacuous(self):
        multi = {"multiple_people", "anonymous_people", "jitter_reorder_duplicates", "swapped_labels"}
        for scenario, row in self.rows.items():
            if scenario in {"crossing_ambiguity", "total_occlusion"}:
                continue
            count = self.FRAMES * (2 if scenario in multi else 1)
            with self.subTest(scenario=scenario):
                self.assertGreaterEqual(row["valid_joints"], count * 25)
                self.assertIsNotNone(row["position_error_mm"])
                self.assertGreater(row["position_error_mm"]["mean"], 0)
                self.assertLess(row["position_error_mm"]["mean"], 20)
                expected_angles = 0 if scenario == "one_view_joint" else count
                self.assertEqual(row["available_angle_samples"], expected_angles)
                if expected_angles:
                    self.assertIsNotNone(row["joint_angle_error_degrees"])
                    self.assertLess(row["joint_angle_error_degrees"]["mean"], 3)

    def test_complementary_occlusion_recovers_all_joints_without_hidden_view_support(self):
        row = self.rows["complementary_occlusion"]
        self.assertEqual(row["valid_joints"], self.FRAMES * 33)
        self.assertEqual(row["unavailable_joints"], 0)
        self.assertEqual(row["recovered_occluded_joints"], self.FRAMES * 16)
        hidden = {"front": {13, 15, 17, 19, 21}, "left": {14, 16, 18, 20, 22},
                  "right": {23, 25, 27}, "high": {24, 26, 28}}
        for payload in self.by_scenario["complementary_occlusion"]:
            self.assertEqual(len(payload["persons"]), 1)
            person = payload["persons"][0]
            self.assertEqual(person["identity_id"], "member-0")
            self.assertTrue(all(person["valid_mask"]))
            for camera, joints in hidden.items():
                for joint in joints:
                    self.assertNotIn(camera, person["joint_support"][joint])
                    self.assertGreaterEqual(len(person["joint_support"][joint]), 2)

    def test_single_view_wrist_is_json_null_and_never_produces_arm_angle(self):
        row = self.rows["one_view_joint"]
        self.assertEqual(row["valid_joints"], self.FRAMES * 32)
        self.assertEqual(row["unavailable_joints"], self.FRAMES)
        self.assertEqual(row["available_angle_samples"], 0)
        self.assertIsNone(row["joint_angle_error_degrees"])
        for payload in self.by_scenario["one_view_joint"]:
            person = payload["persons"][0]
            self.assertFalse(person["valid_mask"][15])
            self.assertIsNone(person["world_points_m"][15])
            self.assertEqual(person["joint_support"][15], [])
            self.assertEqual(sum(person["valid_mask"]), 32)

    def test_multiple_people_remain_distinct_and_anonymous_ids_are_bundle_local(self):
        for payload in self.by_scenario["multiple_people"]:
            self.assertEqual(len(payload["persons"]), 2)
            self.assertEqual({person["identity_id"] for person in payload["persons"]}, {"member-0", "member-1"})
            for person in payload["persons"]:
                self.assertEqual(person["identity_status"], "confirmed")
                self.assertEqual(sum(person["valid_mask"]), 33)
                self.assertEqual(len(person["observation_keys"]), 4)
        identifiers = []
        for payload in self.by_scenario["anonymous_people"]:
            self.assertEqual(len(payload["persons"]), 2)
            for person in payload["persons"]:
                self.assertIsNone(person["identity_id"])
                self.assertEqual(person["identity_status"], "anonymous")
                self.assertEqual(sum(person["valid_mask"]), 33)
                identifiers.append(person["person_id"])
        self.assertEqual(len(set(identifiers)), self.FRAMES * 2)

    def test_exact_crossing_quarantines_every_ambiguous_observation(self):
        row = self.rows["crossing_ambiguity"]
        self.assertEqual(row["confirmed_rows"], 0)
        self.assertGreater(row["ambiguous_rows"], 0)
        central = [payload for payload in self.by_scenario["crossing_ambiguity"]
                   if {key[2] for person in payload["persons"] for key in person["observation_keys"]}
                   == {self.FRAMES // 2}]
        self.assertEqual(len(central), 1)
        people = central[0]["persons"]
        self.assertEqual(sum(len(person["observation_keys"]) for person in people), 8)
        self.assertTrue(all(person["identity_id"] is None for person in people))
        self.assertTrue(all(person["identity_status"] == "ambiguous" for person in people))
        self.assertTrue(all(not any(person["valid_mask"]) for person in people))

    def test_swapped_labels_and_calibration_drift_quarantine_bad_camera(self):
        for scenario, expected_people in (("swapped_labels", 2), ("calibration_drift", 1)):
            for payload in self.by_scenario[scenario]:
                confirmed = [person for person in payload["persons"] if person["identity_status"] == "confirmed"]
                rejected = [person for person in payload["persons"] if person["identity_status"] == "ambiguous"]
                self.assertEqual(len(confirmed), expected_people)
                self.assertEqual(len(rejected), expected_people)
                for person in confirmed:
                    self.assertNotIn("front", {key[0] for key in person["observation_keys"]})
                    self.assertEqual(sum(person["valid_mask"]), 33)
                for person in rejected:
                    self.assertEqual({key[0] for key in person["observation_keys"]}, {"front"})
                    self.assertIsNone(person["identity_id"])
                    self.assertFalse(any(person["valid_mask"]))
        for payload in self.by_scenario["bad_joint_camera"]:
            person = payload["persons"][0]
            self.assertTrue(person["valid_mask"][15])
            self.assertNotIn("front", person["joint_support"][15])
            self.assertGreaterEqual(len(person["joint_support"][15]), 2)

    def test_reused_local_track_does_not_inherit_previous_member(self):
        rows = self.by_scenario["track_id_reuse"]
        self.assertEqual(self.rows["track_id_reuse"]["confirmed_rows"], self.FRAMES // 2)
        for payload in rows:
            self.assertEqual(len(payload["persons"]), 1)
            person = payload["persons"][0]
            sequences = {key[2] for key in person["observation_keys"]}
            self.assertEqual(len(sequences), 1)
            self.assertEqual({key[3] for key in person["observation_keys"]}, {"recycled-track"})
            if next(iter(sequences)) < self.FRAMES // 2:
                self.assertEqual(person["identity_id"], "member-0")
            else:
                self.assertIsNone(person["identity_id"])
                self.assertEqual(person["identity_status"], "anonymous")

    def test_clock_offsets_dropouts_and_duplicate_transport_are_actually_exercised(self):
        clean = self.rows["clean_single"]
        self.assertEqual(clean["sync_stats"]["accepted_frames"], self.FRAMES * 4)
        self.assertEqual(clean["sync_stats"]["partial_bundles"], 0)
        self.assertGreater(self.rows["camera_dropout"]["sync_stats"]["partial_bundles"], 0)
        self.assertGreater(self.rows["mixed_frame_rates"]["sync_stats"]["partial_bundles"], 0)
        jitter = self.rows["jitter_reorder_duplicates"]
        self.assertGreater(jitter["sync_stats"]["duplicate_frames"], 0)
        self.assertGreater(jitter["input_frames_including_replays"], jitter["sync_stats"]["accepted_frames"])
        uncertain = self.rows["uncertain_clock"]
        self.assertEqual(uncertain["sync_stats"]["uncertain_clock_frames"], self.FRAMES)
        self.assertEqual(uncertain["sync_stats"]["accepted_frames"], self.FRAMES * 3)
        for payload in self.by_scenario["uncertain_clock"]:
            self.assertIn("front", payload["missing_cameras"])
            self.assertNotIn("front", {key[0] for person in payload["persons"] for key in person["observation_keys"]})

    def test_total_occlusion_retains_provenance_without_fabricated_geometry(self):
        row = self.rows["total_occlusion"]
        self.assertEqual(row["valid_joints"], 0)
        self.assertEqual(row["available_angle_samples"], 0)
        self.assertIsNone(row["position_error_mm"])
        self.assertIsNone(row["joint_angle_error_degrees"])
        for payload in self.by_scenario["total_occlusion"]:
            self.assertEqual(sum(len(person["observation_keys"]) for person in payload["persons"]), 4)
            for person in payload["persons"]:
                self.assertIsNone(person["identity_id"])
                self.assertFalse(any(person["valid_mask"]))
                self.assertTrue(all(point is None for point in person["world_points_m"]))

    def test_reopened_sqlite_replays_every_packet_without_consuming_observations_twice(self):
        counts = self.report["local_storage"]["counts_added"]
        store = LocalFusionStore.sqlite(self.database)
        self.addCleanup(store.close)
        self.assertEqual(store.counts, counts)
        seen = set()
        for payload in self.payloads:
            wire = json.dumps(payload, allow_nan=False)
            self.assertEqual(decode_fused_packet(wire).to_dict(), payload)
            self.assertEqual(store.store_json(wire), "duplicate")
            for person in payload["persons"]:
                for key in person["observation_keys"]:
                    self.assertNotIn(tuple(key), seen)
                    seen.add(tuple(key))
        self.assertEqual(len(seen), counts["observations"])
        self.assertEqual(store.counts, counts)
        self.assertEqual(counts["packets"], self.FRAMES * len(simulation.SCENARIOS))
        self.assertEqual(counts["persons"], sum(row["fused_person_rows"] for row in self.rows.values()))


class MultiCameraSimulationFailureControls(unittest.TestCase):
    def evaluate(self, transform):
        scene = simulation.make_scene("clean_single", frame_count=4, seed=42)
        original_fuse = simulation.MultiCameraFusion.fuse

        def changed_fuse(fusion, bundle):
            return transform(original_fuse(fusion, bundle))

        store = LocalFusionStore.sqlite(":memory:")
        self.addCleanup(store.close)
        with patch.object(simulation.MultiCameraFusion, "fuse", new=changed_fuse):
            return simulation.evaluate_scene(scene, store)

    def test_runner_fails_when_fusion_returns_identity_without_any_geometry(self):
        def remove_geometry(packet):
            people = tuple(replace(person, world_points=np.full_like(person.world_points, np.nan),
                                   valid_mask=np.zeros_like(person.valid_mask),
                                   joint_support=((),) * len(person.valid_mask)) for person in packet.persons)
            return replace(packet, persons=people)

        row = self.evaluate(remove_geometry)
        self.assertFalse(row["passed"])
        self.assertEqual(row["confirmed_rows"], 4)  # Identity counts alone would pass.
        self.assertFalse(row["checks"]["metric_person_recovered_in_each_bundle"])
        self.assertFalse(row["checks"]["expected_3d_angles_available"])
        self.assertTrue(row["checks"]["all_source_observations_accounted_for"])

    def test_runner_fails_when_fusion_silently_drops_source_observations(self):
        row = self.evaluate(lambda packet: replace(packet, persons=()))
        self.assertFalse(row["passed"])
        self.assertFalse(row["checks"]["all_source_observations_accounted_for"])
        self.assertEqual(row["stored_packets"], 4)

    def test_runner_requires_angles_even_with_enough_other_metric_joints(self):
        def remove_elbow(packet):
            people = []
            for person in packet.persons:
                mask = person.valid_mask.copy()
                mask[13] = False
                support = list(person.joint_support)
                support[13] = ()
                people.append(replace(person, valid_mask=mask, joint_support=tuple(support)))
            return replace(packet, persons=tuple(people))

        row = self.evaluate(remove_elbow)
        self.assertFalse(row["passed"])
        self.assertTrue(row["checks"]["metric_person_recovered_in_each_bundle"])
        self.assertFalse(row["checks"]["expected_3d_angles_available"])

    def test_runner_catches_rare_bad_joints_hidden_by_a_good_mean(self):
        def move_one_joint(packet):
            people = []
            for person in packet.persons:
                points = person.world_points.copy()
                points[30, 0] += .10
                people.append(replace(person, world_points=points))
            return replace(packet, persons=tuple(people))

        row = self.evaluate(move_one_joint)
        self.assertFalse(row["passed"])
        self.assertTrue(row["checks"]["reconstructed_mean_error_below_20_mm"])
        self.assertFalse(row["checks"]["reconstructed_max_error_below_50_mm"])

    def test_invalid_or_empty_runs_fail_before_output_or_calibration(self):
        invalid = (
            {"seeds": ()}, {"seeds": (42, 42)}, {"seeds": (True,)}, {"seeds": (-1,)},
            {"scenarios": ()}, {"scenarios": ("clean_single", "clean_single")},
            {"scenarios": ("unknown",)}, {"frames": 0}, {"frames": True}, {"frames": 4.0},
        )
        with tempfile.TemporaryDirectory(prefix="invalid-multicam-") as directory:
            for index, options in enumerate(invalid):
                output = Path(directory) / f"invalid-{index}"
                with self.subTest(options=options), patch.object(simulation, "estimate_virtual_rig") as calibrate:
                    with self.assertRaises(ValueError):
                        simulation.run_simulation(output, **options)
                    calibrate.assert_not_called()
                    self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
