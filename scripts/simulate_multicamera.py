"""Replay calibrated multi-person camera faults through sync, fusion and storage.

The live camera UI is not connected to this offline pipeline. Synthetic identities
and visibility are simulated detections; ground truth is used only by evaluation.
"""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Mirror_Code"))

from core.camera_calibration import calibrate_camera, calibrate_stereo
from core.multicamera_fusion import MultiCameraFusion
from core.multicamera_store import LocalFusionStore
from core.multicamera_sync import FrameSynchronizer
from core.synthetic_camera import default_camera, generate_chessboard_views, render_chessboard
from core.synthetic_multicamera import SCENARIOS, make_rig, make_scene
from core.synthetic_pose import calculate_joint_angle_3d


def _corners(view):
    found, corners = cv2.findChessboardCornersSB(view.image, (9, 6), cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    grid = corners.reshape(6, 9, 2)
    # Symmetric synthetic board IDs establish ordering only; the detector
    # supplies all measured coordinates used by the calibration solver.
    reference = view.image_points.reshape(-1, 2)
    return min((grid, grid[::-1], grid[:, ::-1], grid[::-1, ::-1]),
               key=lambda p: np.linalg.norm(p.reshape(-1, 2) - reference, axis=1).mean()).reshape(-1, 2).copy()


def estimate_virtual_rig(seed=42):
    model = default_camera()
    mono = [(view, _corners(view)) for view in generate_chessboard_views(count=48, seed=seed)]
    accepted = [(view, corners) for view, corners in mono if corners is not None]
    if len(accepted) < 42:
        raise ValueError("too few rendered calibration boards detected")
    training = accepted[:-6]
    calibration = calibrate_camera([v.object_points for v, _ in training], [p for _, p in training],
                                   model.image_size, min_views=10, max_rms_px=.6)
    truth = make_rig()
    rng = np.random.default_rng(seed + 900)
    views = {camera_id: [] for camera_id in truth}
    for index in range(18):
        board_rotation, _ = cv2.Rodrigues(rng.uniform([-.35, -.35, -.12], [.35, .35, .12]))
        board_center = rng.uniform([-.15, -.15, 2.8], [.15, .15, 3.2])
        board_translation = board_center - board_rotation @ np.array([.32, .20, 0.])
        for camera_index, (camera_id, camera) in enumerate(truth.items()):
            rvec, _ = cv2.Rodrigues(camera.rotation @ board_rotation)
            tvec = camera.rotation @ board_translation + camera.translation_m
            view = render_chessboard(model, rvec, tvec, square_size=.08, noise_std=1,
                                     blur_sigma=.5, seed=seed + index * 4 + camera_index)
            views[camera_id].append((view, _corners(view)))
    extrinsics = {"front": (np.eye(3), np.zeros(3))}
    metrics = {"mono_detected": len(accepted), "mono_training_rms_px": calibration.rms_error_px,
               "stereo_pairs": {}, "stereo_rms_px": {}, "extrinsic_translation_error_mm": {}}
    for camera_id in ("left", "right", "high"):
        pairs = [(left, right) for left, right in zip(views["front"], views[camera_id])
                 if left[1] is not None and right[1] is not None]
        stereo = calibrate_stereo([left[0].object_points for left, _ in pairs],
                                  [left[1] for left, _ in pairs], [right[1] for _, right in pairs],
                                  calibration, calibration, min_views=10, max_rms_px=.8)
        extrinsics[camera_id] = (stereo.rotation, stereo.translation)
        metrics["stereo_pairs"][camera_id] = len(pairs)
        metrics["stereo_rms_px"][camera_id] = stereo.rms_error_px
        metrics["extrinsic_translation_error_mm"][camera_id] = float(np.linalg.norm(stereo.translation - truth[camera_id].translation_m) * 1000)
    metrics["mode"] = "intrinsics and three relative transforms recovered from detected rendered-board corners"
    return calibration, extrinsics, metrics


def _summary(values):
    if not values:
        return None
    data = np.asarray(values)
    return {"mean": float(data.mean()), "p95": float(np.percentile(data, 95)), "max": float(data.max())}


def evaluate_scene(scene, store):
    synchronizer = FrameSynchronizer(tuple(scene.cameras.values()), max_skew_seconds=.012,
                                     reorder_horizon_seconds=.100, max_frame_age_seconds=.5,
                                     max_future_lead_seconds=.005, session_id=f"{scene.scenario}-{uuid4().hex[:12]}")
    fusion = MultiCameraFusion(scene.cameras, max_span_seconds=.012, min_shared_joints=4)
    bundles = []
    for frame in scene.frames:
        bundles.extend(synchronizer.add_frame(frame))
    bundles.extend(synchronizer.flush())
    observations = {(frame.camera_id, frame.stream_epoch, frame.sequence, person.local_track_id): person
                    for frame in scene.frames for person in frame.observations}
    errors, angle_errors, reasons = [], [], Counter()
    seen_frames = set()
    contaminated = identity_errors = frame_reuse = frame_mix = false_support = 0
    people = confirmed = anonymous = ambiguous = valid_joints = missing_joints = recovered_occluded = 0
    one_view_wrist_violations = inherited_identity = 0
    complete_two_person_bundles = reconstructed_single_bundles = lost_observation_bundles = 0
    canonical_roundtrips = store_duplicates = stored = 0
    sample = None
    max_span = 0.
    for bundle in bundles:
        reconstructed_actors = set()
        max_span = max(max_span, bundle.span_seconds)
        for frame in bundle.frames:
            key = (frame.camera_id, frame.stream_epoch, frame.sequence)
            frame_reuse += key in seen_frames
            seen_frames.add(key)
        packet = fusion.fuse(bundle)
        expected_keys = {(frame.camera_id, frame.stream_epoch, frame.sequence, person.local_track_id)
                         for frame in bundle.frames for person in frame.observations}
        actual_keys = [key for person in packet.persons for key in person.observation_keys]
        lost_observation_bundles += set(actual_keys) != expected_keys or len(actual_keys) != len(expected_keys)
        wire = packet.to_json()
        if sample is None and packet.persons:
            sample = packet.to_dict()
        result = store.store_json(wire)
        stored += result == "stored"
        canonical_roundtrips += store.read_packet(packet.bundle_id) == packet.to_dict()
        store_duplicates += store.store_json(wire) == "duplicate"
        for person in packet.persons:
            people += 1
            confirmed += person.identity_status == "confirmed"
            anonymous += person.identity_status == "anonymous"
            ambiguous += person.identity_status == "ambiguous"
            reasons.update(person.reasons)
            source = [scene.truth[key] for key in person.observation_keys]
            actors = {item[0] for item in source}
            frames = {item[1] for item in source}
            contaminated += len(actors) != 1
            frame_mix += len(frames) != 1
            if len(actors) != 1:
                continue
            actor = next(iter(actors))
            if person.identity_id is not None and person.identity_id != scene.actor_identities[actor]:
                identity_errors += 1
            if scene.scenario == "track_id_reuse" and actor == 1 and person.identity_id is not None:
                inherited_identity += 1
            expected = source[0][2]
            if int(person.valid_mask.sum()) >= 25:
                reconstructed_actors.add(actor)
            valid_joints += int(person.valid_mask.sum())
            missing_joints += int((~person.valid_mask).sum())
            if scene.scenario == "one_view_joint" and person.valid_mask[15]:
                one_view_wrist_violations += 1
            errors.extend(np.linalg.norm(person.world_points[person.valid_mask] - expected[person.valid_mask], axis=1) * 1000)
            if person.valid_mask[[11, 13, 15]].all():
                angle_errors.append(abs(calculate_joint_angle_3d(*person.world_points[[11, 13, 15]])
                                        - calculate_joint_angle_3d(*expected[[11, 13, 15]])))
            for joint in np.flatnonzero(person.valid_mask):
                supported = set(person.joint_support[joint])
                actual = {key[0] for key in person.observation_keys if observations[key].confidence[joint] >= .5}
                false_support += len(supported) < 2 or not supported <= actual
                recovered_occluded += any(observations[key].confidence[joint] == 0 for key in person.observation_keys)
        complete_two_person_bundles += reconstructed_actors == {0, 1}
        reconstructed_single_bundles += len(reconstructed_actors) == 1
    metric, angle_metric = _summary(errors), _summary(angle_errors)
    checks = {
        "every_capture_step_emitted": len(bundles) == scene.frame_count,
        "all_accepted_frames_consumed": synchronizer.stats["accepted_frames"] == synchronizer.stats["emitted_frames"],
        "no_mixed_person_groups": contaminated == 0,
        "no_wrong_member_identity": identity_errors == 0,
        "no_source_frame_reused": frame_reuse == 0,
        "no_mixed_capture_frames": frame_mix == 0,
        "synchronization_span_within_12_ms": max_span <= .012 + 1e-12,
        "only_observed_multiview_support": false_support == 0,
        "all_source_observations_accounted_for": lost_observation_bundles == 0,
        "json_roundtrip_and_duplicate_ingest": canonical_roundtrips == stored == store_duplicates == len(bundles),
        "no_invented_single_view_wrist": one_view_wrist_violations == 0,
        "no_identity_inherited_by_reused_track": inherited_identity == 0,
        "reconstructed_mean_error_below_20_mm": metric is None or metric["mean"] < 20,
        "reconstructed_mean_angle_error_below_3_degrees": not angle_errors or float(np.mean(angle_errors)) < 3,
        "reconstructed_p95_error_below_20_mm": metric is None or metric["p95"] < 20,
        "reconstructed_max_error_below_50_mm": metric is None or metric["max"] < 50,
        "reconstructed_p95_angle_error_below_3_degrees": angle_metric is None or angle_metric["p95"] < 3,
        "reconstructed_max_angle_error_below_8_degrees": angle_metric is None or angle_metric["max"] < 8,
    }
    if scene.scenario in {"clean_single", "complementary_occlusion", "one_view_joint", "camera_dropout", "mixed_frame_rates", "bad_joint_camera", "calibration_drift", "uncertain_clock"}:
        checks["known_person_recovered_in_each_bundle"] = confirmed == len(bundles)
        checks["metric_person_recovered_in_each_bundle"] = reconstructed_single_bundles == len(bundles)
    if scene.scenario == "complementary_occlusion":
        checks["occluded_joints_recovered_from_other_views"] = recovered_occluded > 0
    if scene.scenario == "total_occlusion":
        checks["total_occlusion_no_metric_joints"] = valid_joints == 0 and not angle_errors
    if scene.scenario == "one_view_joint":
        checks["single_view_wrist_no_3d_angle"] = not angle_errors
    if scene.scenario == "uncertain_clock":
        checks["uncertain_camera_rejected"] = synchronizer.stats.get("uncertain_clock_frames", 0) == scene.frame_count
    if scene.scenario == "crossing_ambiguity":
        checks["ambiguous_observations_quarantined"] = ambiguous > 0 and confirmed == 0
    if scene.scenario == "track_id_reuse":
        checks["both_track_owners_reconstructed"] = reconstructed_single_bundles == len(bundles)
        checks["only_original_track_owner_identified"] = confirmed == scene.frame_count // 2 and anonymous == scene.frame_count - confirmed
    if scene.scenario in {"multiple_people", "anonymous_people", "jitter_reorder_duplicates", "swapped_labels"}:
        checks["two_distinct_people_reconstructed"] = complete_two_person_bundles == len(bundles)
    if scene.scenario == "swapped_labels":
        checks["good_views_recover_both_identities"] = confirmed == 2 * len(bundles) and ambiguous > 0
    if scene.scenario not in {"total_occlusion", "one_view_joint", "crossing_ambiguity"}:
        expected_angles = len(bundles) * (2 if scene.scenario in {
            "multiple_people", "anonymous_people", "jitter_reorder_duplicates", "swapped_labels"} else 1)
        checks["expected_3d_angles_available"] = len(angle_errors) == expected_angles
    return {
        "scenario": scene.scenario, "input_frames_including_replays": len(scene.frames),
        "bundles": len(bundles), "fused_person_rows": people,
        "confirmed_rows": confirmed, "anonymous_rows": anonymous, "ambiguous_rows": ambiguous,
        "valid_joints": valid_joints, "unavailable_joints": missing_joints,
        "recovered_occluded_joints": recovered_occluded, "position_error_mm": metric,
        "joint_angle_error_degrees": angle_metric, "available_angle_samples": len(angle_errors),
        "mixed_person_groups": contaminated, "wrong_identity_assignments": identity_errors,
        "max_capture_span_ms": max_span * 1000, "sync_stats": synchronizer.stats,
        "reason_counts": dict(reasons), "stored_packets": stored, "duplicate_packet_deliveries": store_duplicates,
        "checks": checks, "passed": all(checks.values()), "sample_packet": sample,
    }


def run_simulation(output, *, seeds=(7, 42, 99), frames=18, scenarios=SCENARIOS,
                   calibrated=True, postgres_dsn=None, plots=True):
    seeds, scenarios = tuple(seeds), tuple(scenarios)
    if (not seeds or any(type(seed) is not int or seed < 0 for seed in seeds)
            or len(set(seeds)) != len(seeds)):
        raise ValueError("seeds must be distinct nonnegative integers")
    if (not scenarios or any(scenario not in SCENARIOS for scenario in scenarios)
            or len(set(scenarios)) != len(scenarios)):
        raise ValueError("at least one distinct known scenario is required")
    if type(frames) is not int or frames < 4:
        raise ValueError("frames must be an integer >= 4")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    calibration, extrinsics, calibration_report = (None, None, {"mode": "ideal known virtual calibration"})
    if calibrated:
        calibration, extrinsics, calibration_report = estimate_virtual_rig()
    run_id = uuid4().hex[:12]
    database = output / f"local_fusion_{run_id}.sqlite3"
    store = LocalFusionStore.postgres(postgres_dsn) if postgres_dsn else LocalFusionStore.sqlite(database)
    initial_counts = store.counts
    results = []
    try:
        for seed in seeds:
            for scenario in scenarios:
                scene = make_scene(scenario, frame_count=frames, seed=seed, calibration=calibration,
                                   extrinsics=extrinsics, namespace=run_id)
                result = evaluate_scene(scene, store)
                result["seed"] = seed
                results.append(result)
        final_counts = store.counts
    finally:
        store.close()
    report = {
        "schema_version": 1, "passed": all(row["passed"] for row in results),
        "seeds": list(seeds), "frames_per_camera_per_scenario": frames,
        "scenario_runs": len(results), "calibration": calibration_report,
        "totals": {key: sum(row[key] for row in results) for key in (
            "input_frames_including_replays", "bundles", "valid_joints", "unavailable_joints",
            "recovered_occluded_joints", "mixed_person_groups", "wrong_identity_assignments",
            "stored_packets", "duplicate_packet_deliveries")},
        "local_storage": {"backend": "postgres" if postgres_dsn else "sqlite",
                          "file": None if postgres_dsn else str(database),
                          "counts_added": {key: final_counts[key] - initial_counts[key] for key in final_counts}},
        "elapsed_seconds": time.monotonic() - started, "results": results,
        "limitations": [
            "Synthetic landmark visibility and identity evidence are supplied; real detector occlusion/face accuracy is not measured.",
            "Unknown or ambiguous people remain unassigned; no temporal identity guess is made when every face is hidden.",
            "A joint seen in only one camera remains unavailable in metric 3D, even when other joints can be reconstructed.",
            "Clock offsets are configured inputs; static or slow scenes cannot prove those offsets are physically correct.",
            "This is an offline tested pipeline and separate local sink; live per-camera workers and exercise history are unchanged.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if plots:
        _plot(report, output)
    return report


def _plot(report, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [row for row in report["results"] if row["seed"] == report["seeds"][0]]
    labels = [row["scenario"].replace("_", " ") for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].barh(x, [row["valid_joints"] for row in rows], label="Reconstructed", color="#2878A8")
    axes[0].barh(x, [row["unavailable_joints"] for row in rows], left=[row["valid_joints"] for row in rows], label="Unavailable / quarantined", color="#C3C8CF")
    axes[0].set_yticks(x, labels)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Joint observations across synchronized bundles", title="Availability is explicit, not filled with guesses")
    axes[0].legend(fontsize=8)
    errors = [row["position_error_mm"]["mean"] if row["position_error_mm"] else 0 for row in rows]
    axes[1].barh(x, errors, color="#208578")
    axes[1].set_yticks(x, labels)
    axes[1].invert_yaxis()
    axes[1].axvline(20, linestyle=":", color="#777777", label="20 mm mean error limit")
    axes[1].set(xlabel="Mean error for available 3D joints (mm)", title="Zero bars with no available joints mean no estimate")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Internal multi-camera occlusion and identity fault replay", fontsize=14)
    figure.savefig(output / "multicamera_results.png", dpi=150)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "multicamera_simulation")
    parser.add_argument("--frames", type=int, default=18)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 42, 99])
    parser.add_argument("--scenarios", choices=SCENARIOS, nargs="+", default=SCENARIOS)
    parser.add_argument("--ideal-calibration", action="store_true", help="Skip raster calibration; use known virtual cameras")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--postgres", action="store_true", help="Use explicit SMART_GYM_FUSION_TEST_PG_DSN disposable DB")
    args = parser.parse_args()
    dsn = os.getenv("SMART_GYM_FUSION_TEST_PG_DSN") if args.postgres else None
    if args.postgres and not dsn:
        parser.error("--postgres requires SMART_GYM_FUSION_TEST_PG_DSN pointing to a disposable test database")
    report = run_simulation(args.output_dir, seeds=args.seeds, frames=args.frames, scenarios=args.scenarios,
                            calibrated=not args.ideal_calibration, postgres_dsn=dsn, plots=not args.no_plots)
    print(json.dumps({"passed": report["passed"], "scenario_runs": report["scenario_runs"],
                      "totals": report["totals"], "failed_checks": [
                          {"scenario": row["scenario"], "seed": row["seed"], "checks": [k for k, value in row["checks"].items() if not value]}
                          for row in report["results"] if not row["passed"]],
                      "report": str(args.output_dir.resolve() / "report.json")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
