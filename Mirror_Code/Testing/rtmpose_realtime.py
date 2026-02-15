#!/usr/bin/env python3
"""Cross-platform real-time pose demo with low-CPU defaults.

This script provides two runtime engines behind a single CLI:
1) `rtmpose` engine: rtmlib + onnxruntime on CPU (works on Linux/macOS/Windows).
2) `rknn` engine: Ultralytics YOLO pose model in `.rknn` format (Rockchip Linux only).

`auto` mode picks RKNN when available, otherwise falls back to CPU RTMPose.
MPS is intentionally not used in this script.
"""

from __future__ import annotations

import argparse
from collections import deque
import os
import platform
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Keep RTMPose checkpoint cache inside the project workspace.
_CACHE_ROOT = Path(__file__).resolve().parent / ".cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("TORCH_HOME", str(_CACHE_ROOT / "rtmlib"))

try:
    from rtmlib import Body, PoseTracker, draw_skeleton as rtmlib_draw_skeleton
except Exception:
    Body = None
    PoseTracker = None
    rtmlib_draw_skeleton = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None


COCO17_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


class RTMPoseCPURuntime:
    """CPU RTMPose runtime via rtmlib PoseTracker."""

    def __init__(self, args: argparse.Namespace) -> None:
        if PoseTracker is None or Body is None:
            raise RuntimeError(
                "rtmlib is not installed. Install dependencies from Testing/requirements.txt"
            )

        self.model = PoseTracker(
            Body,
            det_frequency=max(1, int(args.det_frequency)),
            tracking=not args.no_tracking,
            mode=args.mode,
            backend="onnxruntime",
            device="cpu",
        )
        if hasattr(self.model, "det_model") and hasattr(self.model.det_model, "score_thr"):
            self.model.det_model.score_thr = float(args.det_score_thr)

        self.kpt_thr = float(args.kpt_thr)
        self.smooth_alpha = float(args.smooth_alpha)
        self.infer_every = max(1, int(args.infer_every))
        self._frame_idx = 0
        self._last_keypoints = np.empty((0, 17, 2), dtype=np.float32)
        self._last_scores = np.empty((0, 17), dtype=np.float32)
        self._smoothed_keypoints = None

    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._frame_idx % self.infer_every == 0:
            keypoints, scores = self.model(frame)
            keypoints = np.asarray(keypoints, dtype=np.float32)
            scores = np.asarray(scores, dtype=np.float32)
            self._last_keypoints = keypoints
            self._last_scores = scores
        else:
            keypoints = self._last_keypoints
            scores = self._last_scores

        self._frame_idx += 1

        if isinstance(keypoints, np.ndarray) and keypoints.size > 0:
            if (
                isinstance(self._smoothed_keypoints, np.ndarray)
                and self._smoothed_keypoints.shape == keypoints.shape
                and self.smooth_alpha > 0.0
            ):
                visible_mask = scores > self.kpt_thr
                keypoints = keypoints.copy()
                keypoints[visible_mask] = (
                    self.smooth_alpha * self._smoothed_keypoints[visible_mask]
                    + (1.0 - self.smooth_alpha) * keypoints[visible_mask]
                )
            self._smoothed_keypoints = keypoints.copy()
        else:
            self._smoothed_keypoints = None

        return keypoints, scores


class RKNNPoseRuntime:
    """RKNN runtime via Ultralytics with a `.rknn` pose model."""

    def __init__(self, args: argparse.Namespace) -> None:
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed. Install Testing/requirements.rknn.txt"
            )

        model_path = Path(args.rknn_model).expanduser() if args.rknn_model else None
        if model_path is None or not model_path.exists():
            raise RuntimeError(
                "RKNN model not found. Pass --rknn-model /path/to/model.rknn"
            )
        if model_path.suffix.lower() != ".rknn":
            raise RuntimeError("--rknn-model must point to a .rknn file")

        self.model = YOLO(str(model_path))
        self.conf = float(args.rknn_conf)
        self.iou = float(args.rknn_iou)
        self.imgsz = int(args.rknn_imgsz)
        self.max_det = int(args.rknn_max_det)
        self.kpt_thr = float(args.kpt_thr)
        self.smooth_alpha = float(args.smooth_alpha)
        self.infer_every = max(1, int(args.infer_every))
        self._frame_idx = 0
        self._last_keypoints = np.empty((0, 17, 2), dtype=np.float32)
        self._last_scores = np.empty((0, 17), dtype=np.float32)
        self._smoothed_keypoints = None

    def _infer_once(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        results = self.model.predict(
            source=frame,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            device="cpu",
        )
        if not results:
            return (
                np.empty((0, 17, 2), dtype=np.float32),
                np.empty((0, 17), dtype=np.float32),
            )

        result = results[0]
        if result.keypoints is None or len(result.keypoints) == 0:
            return (
                np.empty((0, 17, 2), dtype=np.float32),
                np.empty((0, 17), dtype=np.float32),
            )

        keypoints_xy = result.keypoints.xy
        keypoints_conf = result.keypoints.conf

        keypoints = keypoints_xy.cpu().numpy().astype(np.float32)
        if keypoints_conf is None:
            scores = np.ones((keypoints.shape[0], keypoints.shape[1]), dtype=np.float32)
        else:
            scores = keypoints_conf.cpu().numpy().astype(np.float32)
        return keypoints, scores

    def infer(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self._frame_idx % self.infer_every == 0:
            keypoints, scores = self._infer_once(frame)
            self._last_keypoints = keypoints
            self._last_scores = scores
        else:
            keypoints = self._last_keypoints
            scores = self._last_scores

        self._frame_idx += 1

        if isinstance(keypoints, np.ndarray) and keypoints.size > 0:
            if (
                isinstance(self._smoothed_keypoints, np.ndarray)
                and self._smoothed_keypoints.shape == keypoints.shape
                and self.smooth_alpha > 0.0
            ):
                visible_mask = scores > self.kpt_thr
                keypoints = keypoints.copy()
                keypoints[visible_mask] = (
                    self.smooth_alpha * self._smoothed_keypoints[visible_mask]
                    + (1.0 - self.smooth_alpha) * keypoints[visible_mask]
                )
            self._smoothed_keypoints = keypoints.copy()
        else:
            self._smoothed_keypoints = None

        return keypoints, scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run low-CPU real-time pose with CPU RTMPose fallback and optional RKNN engine."
        )
    )
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--camera-backend",
        type=str,
        default="auto",
        choices=("auto", "default", "v4l2", "avfoundation"),
        help="Camera backend selection.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="auto",
        choices=("auto", "rtmpose", "rknn"),
        help="Runtime engine.",
    )

    # RTMPose CPU settings.
    parser.add_argument(
        "--mode",
        type=str,
        default="lightweight",
        choices=("lightweight", "balanced", "performance"),
        help="RTMPose preset in rtmlib.",
    )
    parser.add_argument(
        "--det-frequency",
        type=int,
        default=4,
        help="Run person detection every Nth frame in RTMPose mode.",
    )
    parser.add_argument(
        "--det-score-thr",
        type=float,
        default=0.8,
        help="RTMPose detector score threshold.",
    )
    parser.add_argument(
        "--no-tracking",
        action="store_true",
        help="Disable RTMPose tracker if you need lower latency.",
    )

    # Shared controls.
    parser.add_argument(
        "--infer-every",
        type=int,
        default=2,
        help="Run full inference every Nth frame; reuse previous output between calls.",
    )
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=0.35,
        help="Keypoint confidence threshold for drawing.",
    )
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.65,
        help="Temporal smoothing factor for keypoints (0=no smoothing, 0.9=strong).",
    )

    # RKNN settings.
    parser.add_argument(
        "--rknn-model",
        type=str,
        default="",
        help="Path to a pose .rknn model (used by rknn engine).",
    )
    parser.add_argument(
        "--rknn-imgsz",
        type=int,
        default=640,
        help="RKNN model inference image size.",
    )
    parser.add_argument(
        "--rknn-conf",
        type=float,
        default=0.35,
        help="RKNN pose confidence threshold.",
    )
    parser.add_argument(
        "--rknn-iou",
        type=float,
        default=0.5,
        help="RKNN NMS IoU threshold.",
    )
    parser.add_argument(
        "--rknn-max-det",
        type=int,
        default=4,
        help="Maximum detections per frame in RKNN mode.",
    )

    # CPU load controls.
    parser.add_argument("--width", type=int, default=640, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=480, help="Requested camera height.")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=2,
        help="Cap CPU threads for ONNX/OpenMP (0 keeps environment defaults).",
    )
    parser.add_argument(
        "--opencv-threads",
        type=int,
        default=1,
        help="OpenCV thread cap (1 usually lowers CPU spikes).",
    )
    parser.add_argument(
        "--max-fps",
        type=float,
        default=24.0,
        help="Throttle display loop to this FPS (0 disables throttling).",
    )
    parser.add_argument("--show-fps", action="store_true", help="Overlay moving-average FPS.")
    return parser.parse_args()


def _is_rockchip_linux() -> bool:
    if platform.system().lower() != "linux":
        return False

    machine = platform.machine().lower()
    if machine not in {"aarch64", "arm64"}:
        return False

    candidates = [
        Path("/proc/device-tree/compatible"),
        Path("/sys/firmware/devicetree/base/compatible"),
    ]
    for path in candidates:
        try:
            data = path.read_bytes().decode("utf-8", errors="ignore").lower()
            if "rockchip" in data:
                return True
        except Exception:
            continue

    return "rockchip" in platform.platform().lower()


def _configure_cpu_runtime(cpu_threads: int, opencv_threads: int) -> None:
    if cpu_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(cpu_threads)
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        os.environ.setdefault("MKL_NUM_THREADS", str(cpu_threads))

    if opencv_threads > 0:
        try:
            cv2.setNumThreads(opencv_threads)
        except Exception:
            pass


def _open_camera(camera_id: int, backend: str) -> cv2.VideoCapture:
    backend = backend.lower()
    if backend == "v4l2":
        return cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    if backend == "avfoundation":
        return cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
    if backend == "default":
        return cv2.VideoCapture(camera_id)

    # auto
    if platform.system().lower() == "linux":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        if cap.isOpened():
            return cap
        cap.release()
    if platform.system().lower() == "darwin":
        cap = cv2.VideoCapture(camera_id, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(camera_id)


def _draw_pose_basic(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    kpt_thr: float,
) -> np.ndarray:
    if not isinstance(keypoints, np.ndarray) or keypoints.size == 0:
        return frame

    if keypoints.ndim == 2:
        keypoints = keypoints[None, ...]
    if not isinstance(scores, np.ndarray) or scores.size == 0:
        scores = np.ones((keypoints.shape[0], keypoints.shape[1]), dtype=np.float32)
    elif scores.ndim == 1 and keypoints.shape[0] == 1:
        scores = scores[None, ...]

    for person_idx in range(keypoints.shape[0]):
        xy = keypoints[person_idx]
        conf = scores[person_idx]

        for a, b in COCO17_EDGES:
            if conf[a] < kpt_thr or conf[b] < kpt_thr:
                continue
            xa, ya = int(xy[a, 0]), int(xy[a, 1])
            xb, yb = int(xy[b, 0]), int(xy[b, 1])
            cv2.line(frame, (xa, ya), (xb, yb), (0, 255, 255), 2, cv2.LINE_AA)

        for j in range(xy.shape[0]):
            if conf[j] < kpt_thr:
                continue
            xj, yj = int(xy[j, 0]), int(xy[j, 1])
            cv2.circle(frame, (xj, yj), 3, (0, 255, 0), -1, cv2.LINE_AA)

    return frame


def _draw_pose(
    frame: np.ndarray,
    keypoints: np.ndarray,
    scores: np.ndarray,
    kpt_thr: float,
) -> np.ndarray:
    if rtmlib_draw_skeleton is not None:
        try:
            return rtmlib_draw_skeleton(
                frame,
                keypoints,
                scores,
                openpose_skeleton=False,
                kpt_thr=kpt_thr,
                radius=3,
                line_width=2,
            )
        except Exception:
            pass
    return _draw_pose_basic(frame, keypoints, scores, kpt_thr)


def _build_runtime(args: argparse.Namespace):
    runtime_errors = []

    should_try_rknn = False
    if args.engine == "rknn":
        should_try_rknn = True
    elif args.engine == "auto":
        # Try RKNN automatically only on Rockchip Linux when model path is provided.
        should_try_rknn = bool(args.rknn_model) and _is_rockchip_linux()

    if should_try_rknn:
        try:
            return RKNNPoseRuntime(args), "rknn"
        except Exception as exc:
            if args.engine == "rknn":
                raise
            runtime_errors.append(f"RKNN unavailable: {exc}")

    try:
        return RTMPoseCPURuntime(args), "rtmpose-cpu"
    except Exception as exc:
        if runtime_errors:
            details = " | ".join(runtime_errors)
            raise RuntimeError(f"Could not initialize runtime ({details}); RTMPose error: {exc}")
        raise


def main() -> int:
    args = parse_args()
    args.det_frequency = max(1, args.det_frequency)
    args.infer_every = max(1, args.infer_every)
    args.smooth_alpha = min(max(args.smooth_alpha, 0.0), 0.95)

    _configure_cpu_runtime(int(args.cpu_threads), int(args.opencv_threads))

    try:
        runtime, runtime_name = _build_runtime(args)
    except Exception as exc:
        print(f"[ERROR] Failed to initialize pose runtime: {exc}", file=sys.stderr)
        return 1

    cap = _open_camera(args.camera_id, args.camera_backend)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera index {args.camera_id}.", file=sys.stderr)
        return 1

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.height))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    window_name = "Pose Realtime (CPU optimized) - press q to quit"
    frame_times = deque(maxlen=60)
    last_frame_time = None

    target_frame_time = 0.0
    if args.max_fps and args.max_fps > 0:
        target_frame_time = 1.0 / float(args.max_fps)

    print(f"[INFO] Runtime engine: {runtime_name}")

    try:
        while True:
            loop_start = time.perf_counter()

            ok, frame = cap.read()
            if not ok:
                print("[WARN] Camera frame read failed, stopping.", file=sys.stderr)
                break

            try:
                keypoints, scores = runtime.infer(frame)
            except Exception as exc:
                print(f"[WARN] Inference failed this frame: {exc}", file=sys.stderr)
                keypoints = np.empty((0, 17, 2), dtype=np.float32)
                scores = np.empty((0, 17), dtype=np.float32)

            frame = _draw_pose(frame, keypoints, scores, float(args.kpt_thr))

            if args.show_fps:
                now = time.perf_counter()
                if last_frame_time is not None:
                    frame_times.append(now - last_frame_time)
                last_frame_time = now
                fps = len(frame_times) / max(sum(frame_times), 1e-6) if frame_times else 0.0
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f} | {runtime_name}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame,
                    runtime_name,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (220, 220, 220),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if target_frame_time > 0.0:
                elapsed = time.perf_counter() - loop_start
                if elapsed < target_frame_time:
                    time.sleep(target_frame_time - elapsed)
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
