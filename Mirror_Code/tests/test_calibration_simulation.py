"""End-to-end image detection -> calibration -> held-out metric reconstruction."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "simulate_3d_calibration.py"
SPEC = importlib.util.spec_from_file_location("simulate_3d_calibration", SCRIPT)
simulation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(simulation)


class CalibrationSimulationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="calibration-e2e-")
        cls.addClassCleanup(cls.directory.cleanup)
        cls.report = simulation.run_simulation(Path(cls.directory.name), seed=42, plots=False)

    def test_detected_image_calibration_meets_held_out_quality_targets(self):
        report = self.report
        self.assertTrue(report["passed"], report["checks"])
        camera = report["camera_calibration"]
        self.assertEqual(camera["detected_views"], 48)
        self.assertGreaterEqual(camera["training_views"], 10)
        self.assertEqual(camera["held_out_views"], 6)
        self.assertLess(camera["held_out_corrected_ray_error_px"]["p95"], 2.0)
        self.assertLess(camera["held_out_corrected_ray_error_px"]["mean"], camera["held_out_raw_lens_error_px"]["mean"])
        stored = json.loads((Path(self.directory.name) / "report.json").read_text())
        self.assertEqual(stored, report)
        self.assertTrue((Path(self.directory.name) / "detected_checkerboards.png").is_file())

    def test_single_camera_angle_remains_biased_after_lens_calibration(self):
        rows = self.report["pose_benchmarks"]
        edge_on = next(row for row in rows if row["plane_yaw_degrees"] == 80 and row["landmark_noise_std_px"] == 0)
        self.assertGreater(edge_on["lens_corrected_2d_angle_error_degrees"]["mean"], 20)
        self.assertLess(edge_on["stereo_angle_error_degrees"]["mean"], 0.2)

    def test_baseline_tradeoff_exposes_a_rig_that_misses_accuracy_target(self):
        self.assertFalse(self.report["narrow_baseline_comparison"]["meets_3_degree_mean_angle_target_at_0_5_px_noise"])
        self.assertTrue(self.report["checks"]["stereo_mean_angle_error_below_3_degrees_at_0_5_px_noise"])
        narrow = self.report["narrow_baseline_comparison"]["pose_benchmarks"]
        wide = self.report["pose_benchmarks"]
        for yaw in (60, 80):
            a = next(row for row in narrow if row["plane_yaw_degrees"] == yaw and row["landmark_noise_std_px"] == 0.5)
            b = next(row for row in wide if row["plane_yaw_degrees"] == yaw and row["landmark_noise_std_px"] == 0.5)
            self.assertGreater(a["stereo_angle_error_degrees"]["mean"], b["stereo_angle_error_degrees"]["mean"])
