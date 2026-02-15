#!/usr/bin/env python3
"""Linux edge-device RTMPose demo (CPU-only, real-time body landmarks)."""

from __future__ import annotations

import argparse
from collections import deque
import os
import sys
import time
from pathlib import Path

# Keep model cache in project folder for predictable behavior on edge devices.
_CACHE_ROOT = Path(__file__).resolve().parent / ".cache_edge_linux"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("TORCH_HOME", str(_CACHE_ROOT / "rtmlib"))

import cv2
import numpy as np
from rtmlib import Body, PoseTracker, draw_skeleton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RTMPose on Linux edge (CPU only) and draw body landmarks."
    )
    parser.add_argument(
        "--camera-device",
        type=str,
        default="/dev/video0",
        help="Linux camera device path (example: /dev/video0) or numeric index.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="lightweight",
        choices=("lightweight", "balanced"),
        help="Body model preset. Lightweight is recommended for edge devices.",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=2,
        help="Limit CPU threads used by ONNXRuntime/OpenMP (set 0 for system default).",
    )
    parser.add_argument(
        "--det-frequency",
        type=int,
        default=3,
        help="Run person detection every Nth frame and track between detections.",
    )
    parser.add_argument(
        "--infer-every",
        type=int,
        default=2,
        help="Run full pose inference every Nth frame; reuse previous keypoints in-between.",
    )
    parser.add_argument(
        "--det-score-thr",
        type=float,
        default=0.80,
        help="Detector confidence threshold (higher can reduce CPU).",
    )
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=0.35,
        help="Landmark confidence threshold.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.65,
        help="Temporal smoothing (0=no smoothing, 0.9=strong smoothing).",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable tracker for higher speed at the cost of more jitter.",
    )
    parser.add_argument("--width", type=int, default=640, help="Requested frame width.")
    parser.add_argument("--height", type=int, default=480, help="Requested frame height.")
    parser.add_argument("--show-fps", action="store_true", help="Show moving-average FPS.")
    return parser.parse_args()


def _open_linux_camera(camera_device: str) -> cv2.VideoCapture:
    # Prefer V4L2 on Linux edge systems.
    if camera_device.isdigit():
        return cv2.VideoCapture(int(camera_device), cv2.CAP_V4L2)
    return cv2.VideoCapture(camera_device, cv2.CAP_V4L2)


def main() -> int:
    args = parse_args()
    args.det_frequency = max(1, args.det_frequency)
    args.infer_every = max(1, args.infer_every)
    args.smooth_alpha = min(max(args.smooth_alpha, 0.0), 0.95)

    if args.cpu_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))

    try:
        model = PoseTracker(
            Body,
            det_frequency=args.det_frequency,
            tracking=not args.no_tracking,
            mode=args.mode,
            backend="onnxruntime",
            device="cpu",
        )
        if hasattr(model, "det_model") and hasattr(model.det_model, "score_thr"):
            model.det_model.score_thr = args.det_score_thr
    except Exception as exc:
        print(f"[ERROR] Failed to initialize edge RTMPose model: {exc}", file=sys.stderr)
        return 1

    cap = _open_linux_camera(args.camera_device)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera source: {args.camera_device}", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window_name = "RTMPose Edge Linux (press q to quit)"
    frame_times = deque(maxlen=30)
    last_frame_time = None
    last_keypoints = []
    last_scores = []
    smoothed_keypoints = None
    frame_idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[WARN] Camera frame read failed, stopping.", file=sys.stderr)
                break

            if frame_idx % args.infer_every == 0:
                keypoints, scores = model(frame)
                last_keypoints, last_scores = keypoints, scores
            else:
                keypoints, scores = last_keypoints, last_scores

            if isinstance(keypoints, np.ndarray) and keypoints.size > 0:
                if (
                    isinstance(smoothed_keypoints, np.ndarray)
                    and smoothed_keypoints.shape == keypoints.shape
                    and args.smooth_alpha > 0.0
                ):
                    visible_mask = scores > args.kpt_thr
                    keypoints = keypoints.copy()
                    keypoints[visible_mask] = (
                        args.smooth_alpha * smoothed_keypoints[visible_mask]
                        + (1.0 - args.smooth_alpha) * keypoints[visible_mask]
                    )
                smoothed_keypoints = keypoints.copy()
            else:
                smoothed_keypoints = None

            frame = draw_skeleton(
                frame,
                keypoints,
                scores,
                openpose_skeleton=False,
                kpt_thr=args.kpt_thr,
                radius=3,
                line_width=2,
            )

            if args.show_fps:
                now = time.perf_counter()
                if last_frame_time is not None:
                    frame_times.append(now - last_frame_time)
                last_frame_time = now
                fps = len(frame_times) / max(sum(frame_times), 1e-6) if frame_times else 0.0
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            frame_idx += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
