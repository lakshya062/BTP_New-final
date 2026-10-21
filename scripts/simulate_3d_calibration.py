"""Measure camera/stereo calibration on rendered boards and known 3D poses.

No camera, database, network, or pose-model inference is used. The program exits
nonzero if a quality check fails and writes a JSON report plus visual evidence.
"""

import argparse
import json
import os
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Mirror_Code"))

from core.camera_calibration import (
    CalibrationError, calibrate_camera, calibrate_stereo, triangulate_stereo_points,
)
from core.synthetic_camera import default_camera, generate_chessboard_views, render_chessboard
from core.synthetic_pose import (
    calculate_joint_angle_3d, generate_arm_trajectory, project_pose, undistort_pose,
)
from core.utils import calculate_joint_angle

PATTERN = (9, 6)


def _statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Error statistics require finite nonempty observations")
    return {"mean": float(values.mean()), "p95": float(np.percentile(values, 95)), "max": float(values.max())}


def _detected_corners(view):
    found, corners = cv2.findChessboardCornersSB(view.image, PATTERN, cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not found:
        return None
    # A plain chessboard is symmetric. In this synthetic fixture only, known
    # corner IDs resolve orientation; measured coordinates still come from the
    # raster detector. Physical stereo capture needs consistently marked IDs.
    grid = corners.reshape(PATTERN[1], PATTERN[0], 2)
    candidates = (grid, grid[::-1], grid[:, ::-1], grid[::-1, ::-1])
    reference = view.image_points.reshape(-1, 2)
    return min(candidates, key=lambda p: np.linalg.norm(p.reshape(-1, 2) - reference, axis=1).mean()).reshape(-1, 2).copy()


def _calibrate_rendered_camera(camera, views, output):
    detected = [(view, _detected_corners(view)) for view in views]
    accepted = [(view, corners) for view, corners in detected if corners is not None]
    if len(accepted) < 16:
        raise CalibrationError(f"Only {len(accepted)} rendered boards detected; at least 16 required")
    split = len(accepted) - 6
    training, held_out = accepted[:split], accepted[split:]
    calibration = calibrate_camera(
        [v.object_points for v, _ in training], [c for _, c in training], camera.image_size,
        min_views=10, max_rms_px=0.6,
    )
    true_focal = np.diag(camera.camera_matrix)[:2]
    focal_error = np.abs(np.diag(calibration.camera_matrix)[:2] / true_focal - 1) * 100
    principal_error = np.linalg.norm(calibration.camera_matrix[:2, 2] - camera.camera_matrix[:2, 2])
    raw_errors, corrected_errors, corner_errors = [], [], []
    for view, corners in held_out:
        rotation, _ = cv2.Rodrigues(view.rotation_vector)
        points_camera = view.object_points @ rotation.T + view.translation_vector.reshape(3)
        true_rays = points_camera[:, :2] / points_camera[:, 2:]
        # Evaluate new board poses against known rays, without refitting their
        # extrinsics. Low training reprojection RMS alone can hide bad intrinsics.
        raw_rays = (corners - camera.camera_matrix[:2, 2]) / true_focal
        recovered_rays = undistort_pose(corners, calibration.camera_matrix, calibration.distortion_coefficients)
        raw_errors.extend(np.linalg.norm((raw_rays - true_rays) * true_focal, axis=1))
        corrected_errors.extend(np.linalg.norm((recovered_rays - true_rays) * true_focal, axis=1))
        corner_errors.extend(np.linalg.norm(corners - view.image_points.reshape(-1, 2), axis=1))

    tiles = []
    for view, corners in accepted[:6]:
        image = cv2.cvtColor(view.image, cv2.COLOR_GRAY2BGR)
        cv2.drawChessboardCorners(image, PATTERN, corners.astype(np.float32).reshape(-1, 1, 2), True)
        tiles.append(cv2.resize(image, (480, 360)))
    cv2.imwrite(str(output / "detected_checkerboards.png"), np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])]))
    details = {
        "generated_views": len(views), "detected_views": len(accepted),
        "training_views": len(training), "held_out_views": len(held_out),
        "image_size": list(camera.image_size), "square_size_m": 0.03,
        "true_camera_matrix": camera.camera_matrix.tolist(), "estimated_camera_matrix": calibration.camera_matrix.tolist(),
        "true_distortion": camera.distortion.tolist(), "estimated_distortion": calibration.distortion_coefficients.tolist(),
        "training_rms_px": calibration.rms_error_px, "focal_error_percent_xy": focal_error.tolist(),
        "principal_point_error_px": float(principal_error),
        "held_out_detector_error_px": _statistics(corner_errors),
        "held_out_raw_lens_error_px": _statistics(raw_errors),
        "held_out_corrected_ray_error_px": _statistics(corrected_errors),
    }
    return calibration, details


def _calibrate_rendered_stereo(camera, calibration, seed, baseline_m=0.7):
    generator = np.random.default_rng(seed)
    true_rotation, _ = cv2.Rodrigues(np.array([0.0, 0.025, 0.0]))
    true_translation = np.array([-baseline_m, 0.005, 0.01])
    objects, left, right, detector_errors = [], [], [], []
    board_center = np.array([0.12, 0.075, 0.0])
    for index in range(18):
        rvec = generator.uniform([-0.5, -0.5, -0.18], [0.5, 0.5, 0.18])
        rotation, _ = cv2.Rodrigues(rvec)
        # Wider rigs need a more distant board to remain inside both images.
        center = generator.uniform(
            [baseline_m / 2 - 0.14, -0.13, 0.85 + max(0, baseline_m - 0.35)],
            [baseline_m / 2 + 0.14, 0.13, 1.45 + max(0, baseline_m - 0.35)],
        )
        tvec = center - rotation @ board_center
        right_rvec, _ = cv2.Rodrigues(true_rotation @ rotation)
        right_tvec = true_rotation @ tvec + true_translation
        left_view = render_chessboard(camera, rvec, tvec, noise_std=1.0, blur_sigma=0.5, seed=seed + index)
        right_view = render_chessboard(camera, right_rvec, right_tvec, noise_std=1.0, blur_sigma=0.5, seed=seed + 100 + index)
        left_corners, right_corners = _detected_corners(left_view), _detected_corners(right_view)
        if left_corners is None or right_corners is None:
            continue
        objects.append(left_view.object_points)
        left.append(left_corners)
        right.append(right_corners)
        detector_errors.extend(np.linalg.norm(left_corners - left_view.image_points.reshape(-1, 2), axis=1))
        detector_errors.extend(np.linalg.norm(right_corners - right_view.image_points.reshape(-1, 2), axis=1))
    stereo = calibrate_stereo(objects, left, right, calibration, calibration, min_views=10, max_rms_px=0.8)
    rotation_error = np.degrees(np.arccos(np.clip((np.trace(stereo.rotation @ true_rotation.T) - 1) / 2, -1, 1)))
    details = {
        "detected_pairs": len(objects), "training_rms_px": stereo.rms_error_px,
        "true_baseline_m": float(np.linalg.norm(true_translation)), "estimated_baseline_m": float(np.linalg.norm(stereo.translation)),
        "translation_error_mm": float(np.linalg.norm(stereo.translation - true_translation) * 1000),
        "rotation_error_degrees": float(rotation_error),
        "true_right_from_left_rotation": true_rotation.tolist(), "true_right_from_left_translation_m": true_translation.tolist(),
        "estimated_right_from_left_rotation": stereo.rotation.tolist(), "estimated_right_from_left_translation_m": stereo.translation.tolist(),
        "detector_error_px": _statistics(detector_errors),
        "intrinsics_note": "Two identical virtual cameras share the recovered intrinsics; extrinsics are estimated from detected raster corners.",
    }
    return stereo, true_rotation, true_translation, details


def _pose_benchmarks(camera, calibration, stereo, true_rotation, true_translation, seed):
    generator = np.random.default_rng(seed)
    angles = np.linspace(35, 165, 66)
    rows = []
    for yaw in (0, 30, 60, 80):
        points = generate_arm_trajectory(angles, plane_yaw_degrees=yaw) + np.array([-true_translation[0] / 2, 0, 3.0])
        left_clean = project_pose(points, camera.camera_matrix, camera.distortion)
        rvec, _ = cv2.Rodrigues(true_rotation)
        right_clean = project_pose(points, camera.camera_matrix, camera.distortion, rvec, true_translation)
        ideal_rays = points[..., :2] / points[..., 2:]
        ideal_angles = np.array([calculate_joint_angle(*pose) for pose in ideal_rays])
        for noise in (0.0, 0.25, 0.5, 1.0):
            left = left_clean + generator.normal(0, noise, left_clean.shape)
            right = right_clean + generator.normal(0, noise, right_clean.shape)
            raw_angles = np.array([calculate_joint_angle(*pose) for pose in left])
            corrected = undistort_pose(left, calibration.camera_matrix, calibration.distortion_coefficients)
            corrected_angles = np.array([calculate_joint_angle(*pose) for pose in corrected])
            recovered = triangulate_stereo_points(
                left.reshape(-1, 2), right.reshape(-1, 2), calibration, calibration,
                stereo.rotation, stereo.translation, max_reprojection_error_px=5.0,
            ).reshape(points.shape)
            reconstructed_angles = np.array([calculate_joint_angle_3d(*pose) for pose in recovered])
            rows.append({
                "plane_yaw_degrees": yaw, "landmark_noise_std_px": noise,
                "raw_2d_angle_error_degrees": _statistics(np.abs(raw_angles - angles)),
                "lens_corrected_2d_angle_error_degrees": _statistics(np.abs(corrected_angles - angles)),
                "perfect_lens_2d_angle_error_degrees": _statistics(np.abs(ideal_angles - angles)),
                "stereo_angle_error_degrees": _statistics(np.abs(reconstructed_angles - angles)),
                "stereo_position_error_mm": _statistics(np.linalg.norm(recovered - points, axis=-1) * 1000),
            })
    return rows


def _failure_checks(camera, calibration, stereo):
    points = generate_arm_trajectory([60.0]) + np.array([0.175, 0, 3.0])
    left = project_pose(points, camera.camera_matrix, camera.distortion).reshape(-1, 2)
    right = project_pose(points, camera.camera_matrix, camera.distortion, translation_vector=[-0.35, 0, 0]).reshape(-1, 2)
    checks = {}
    for name, operation in {
        "zero_baseline_rejected": lambda: triangulate_stereo_points(left, right, calibration, calibration, np.eye(3), np.zeros(3)),
        "mismatched_correspondence_rejected": lambda: triangulate_stereo_points(left, right + [0, 40], calibration, calibration, np.eye(3), [-0.35, 0, 0]),
        "insufficient_calibration_views_rejected": lambda: calibrate_camera([], [], camera.image_size),
        "parallel_rays_rejected": lambda: triangulate_stereo_points(left, left, calibration, calibration, np.eye(3), [-0.35, 0, 0]),
        "nonfinite_observation_rejected": lambda: triangulate_stereo_points(left * np.nan, right, calibration, calibration, stereo.rotation, stereo.translation),
    }.items():
        try:
            operation()
        except CalibrationError:
            checks[name] = True
        else:
            checks[name] = False
    return checks


def _plot_report(report, output):
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.3), constrained_layout=True)
    rows = report["pose_benchmarks"]
    colors = {0: "#275DAD", 30: "#16817A", 60: "#C08316", 80: "#BA4545"}
    clean = [row for row in rows if row["landmark_noise_std_px"] == 0]
    x = np.arange(len(clean))
    axes[0].bar(x - 0.2, [r["lens_corrected_2d_angle_error_degrees"]["mean"] for r in clean], 0.4, label="Calibrated single camera", color="#C08316")
    axes[0].bar(x + 0.2, [r["stereo_angle_error_degrees"]["mean"] for r in clean], 0.4, label="Calibrated stereo", color="#275DAD")
    axes[0].set_xticks(x, [str(r["plane_yaw_degrees"]) for r in clean])
    axes[0].set(xlabel="Arm plane yaw (degrees)", ylabel="Mean joint-angle error (degrees)", title="Viewing angle still matters")
    axes[0].legend(fontsize=8)
    for yaw, color in colors.items():
        series = [r for r in rows if r["plane_yaw_degrees"] == yaw]
        noise = [r["landmark_noise_std_px"] for r in series]
        axes[1].plot(noise, [r["stereo_position_error_mm"]["mean"] for r in series], "o-", color=color, label=f"{yaw}° yaw")
        axes[2].plot(noise, [r["stereo_angle_error_degrees"]["mean"] for r in series], "o-", color=color)
    axes[1].set(xlabel="Landmark noise (pixel standard deviation)", ylabel="Mean 3D position error (mm)", title="Stereo position accuracy")
    axes[1].legend(fontsize=8)
    narrow = report["narrow_baseline_comparison"]["pose_benchmarks"]
    narrow_worst_yaw = [r for r in narrow if r["plane_yaw_degrees"] == 80]
    axes[2].plot([r["landmark_noise_std_px"] for r in narrow_worst_yaw],
                 [r["stereo_angle_error_degrees"]["mean"] for r in narrow_worst_yaw],
                 "--", color="#BA4545", alpha=0.65, label="35 cm rig, 80° yaw")
    axes[2].axhline(3, color="#777777", linewidth=0.8, linestyle=":", label="3° mean-error target")
    axes[2].set(xlabel="Landmark noise (pixel standard deviation)", ylabel="Mean joint-angle error (degrees)", title="Stereo accuracy: 70 cm vs 35 cm rig")
    axes[2].legend(fontsize=7)
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle("Synthetic calibration validation · measured raster corners + held-out 3D motion", fontsize=13)
    figure.savefig(output / "calibration_results.png", dpi=160)
    plt.close(figure)


def run_simulation(output, seed=42, *, plots=True, view_count=48):
    if isinstance(view_count, bool) or not isinstance(view_count, int) or view_count < 16:
        raise ValueError("view_count must be an integer >= 16 (six views are held out)")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    camera = default_camera()
    views = generate_chessboard_views(count=view_count, camera=camera, noise_std=1.0, blur_sigma=0.5, seed=seed)
    calibration, camera_details = _calibrate_rendered_camera(camera, views, output)
    stereo, true_rotation, true_translation, stereo_details = _calibrate_rendered_stereo(camera, calibration, seed + 1)
    pose_rows = _pose_benchmarks(camera, calibration, stereo, true_rotation, true_translation, seed + 2)
    narrow_stereo, narrow_rotation, narrow_translation, narrow_details = _calibrate_rendered_stereo(
        camera, calibration, seed + 1, baseline_m=0.35,
    )
    narrow_rows = _pose_benchmarks(camera, calibration, narrow_stereo, narrow_rotation, narrow_translation, seed + 2)
    checks = _failure_checks(camera, calibration, stereo)
    checks.update({
        "focal_length_error_below_2_percent": max(camera_details["focal_error_percent_xy"]) < 2.0,
        "held_out_ray_error_p95_below_2_px": camera_details["held_out_corrected_ray_error_px"]["p95"] < 2.0,
        "stereo_translation_error_below_5_mm": stereo_details["translation_error_mm"] < 5.0,
        "stereo_rotation_error_below_0_5_degrees": stereo_details["rotation_error_degrees"] < 0.5,
        "stereo_mean_position_error_below_20_mm_at_0_5_px_noise": all(
            r["stereo_position_error_mm"]["mean"] < 20 for r in pose_rows if r["landmark_noise_std_px"] == 0.5),
        "stereo_mean_angle_error_below_3_degrees_at_0_5_px_noise": all(
            r["stereo_angle_error_degrees"]["mean"] < 3 for r in pose_rows if r["landmark_noise_std_px"] == 0.5),
    })
    report = {
        "schema_version": 1, "seed": seed, "passed": all(checks.values()),
        "opencv_version": cv2.__version__, "numpy_version": np.__version__,
        "camera_calibration": camera_details, "stereo_calibration": stereo_details,
        "pose_benchmarks": pose_rows, "checks": checks,
        "narrow_baseline_comparison": {
            "stereo_calibration": narrow_details, "pose_benchmarks": narrow_rows,
            "meets_3_degree_mean_angle_target_at_0_5_px_noise": all(
                r["stereo_angle_error_degrees"]["mean"] < 3 for r in narrow_rows if r["landmark_noise_std_px"] == 0.5),
        },
        "scope": "Offline simulation only. Live exercise analysis remains 2D; no physical camera has been calibrated by this run.",
        "limitations": [
            "Known synthetic corner IDs resolve chessboard symmetry; real stereo capture needs matched IDs and synchronized frames.",
            "Arm landmarks are exact correspondences with added Gaussian pixel noise, not detections by MediaPipe.",
            "Real optics, rolling shutter, exposure, occlusion, and pose-model bias require hardware/recorded-video validation.",
            "Reported error is for this camera model, 0.70 m main baseline / 0.35 m comparison, 3 m subject distance, and specified noise, not an accuracy guarantee.",
        ],
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    if plots:
        _plot_report(report, output)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "calibration_simulation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--views", type=int, default=48, help="Rendered mono board views; six are held out (default: 48)")
    parser.add_argument("--no-plots", action="store_true")
    arguments = parser.parse_args()
    report = run_simulation(arguments.output_dir, arguments.seed, plots=not arguments.no_plots, view_count=arguments.views)
    print(json.dumps({"passed": report["passed"], "checks": report["checks"], "report": str(arguments.output_dir.resolve() / "report.json")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
