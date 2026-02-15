#!/usr/bin/env python3
"""Real-time OpenPose (OpenCV DNN) webcam demo."""

from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import threading
import time
from urllib.request import urlretrieve

import cv2
import numpy as np


COCO_KEYPOINTS = [
    "Nose",
    "Neck",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
    "REye",
    "LEye",
    "REar",
    "LEar",
]

COCO_PAIRS = (
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
)

DEFAULT_PROTO_URLS = (
    "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/dnn/openpose_pose_coco.prototxt",
    "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/openpose_pose_coco.prototxt",
    "https://huggingface.co/camenduru/openpose/resolve/main/models/pose/coco/pose_deploy_linevec.prototxt?download=true",
)

DEFAULT_WEIGHTS_URLS = (
    "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel",
    "https://huggingface.co/camenduru/openpose/resolve/main/models/pose/coco/pose_iter_440000.caffemodel?download=true",
)


class FrameGrabber:
    """Background frame reader that keeps the latest frame only."""

    def __init__(self, source: int, width: int, height: int) -> None:
        self._source = source
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        self._latest: np.ndarray | None = None
        self._seq = -1

    def start(self) -> None:
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self._source}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._cap is not None
        seq = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest = frame
                self._seq = seq
            seq += 1

    def read_latest(self) -> tuple[bool, np.ndarray | None, int]:
        with self._lock:
            if self._latest is None:
                return False, None, self._seq
            return True, self._latest.copy(), self._seq

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()


def _download_with_fallback(
    file_path: Path,
    urls: list[str],
    min_bytes: int,
) -> None:
    if file_path.exists() and file_path.stat().st_size >= min_bytes:
        return
    if file_path.exists() and file_path.stat().st_size < min_bytes:
        file_path.unlink(missing_ok=True)

    file_path.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in urls:
        try:
            print(f"[openpose] Downloading {file_path.name} from {url}")
            urlretrieve(url, str(file_path))
            size = file_path.stat().st_size if file_path.exists() else 0
            if size < min_bytes:
                raise RuntimeError(
                    f"Downloaded file too small ({size} bytes, expected >= {min_bytes})."
                )
            print(f"[openpose] Downloaded {file_path.name} ({size} bytes)")
            return
        except Exception as exc:
            errors.append(f"{url} -> {exc}")
            file_path.unlink(missing_ok=True)

    err_text = "\n".join(errors)
    raise RuntimeError(
        f"Failed to download '{file_path.name}' from all configured sources:\n{err_text}"
    )


def _resolve_model_files(
    model_dir: Path,
    proto_url: str | None,
    weights_url: str | None,
) -> tuple[Path, Path]:
    proto = model_dir / "pose_deploy_linevec.prototxt"
    weights = model_dir / "pose_iter_440000.caffemodel"

    proto_urls = [proto_url] if proto_url else []
    proto_urls.extend(DEFAULT_PROTO_URLS)
    weights_urls = [weights_url] if weights_url else []
    weights_urls.extend(DEFAULT_WEIGHTS_URLS)

    # De-duplicate while preserving order.
    proto_urls = list(dict.fromkeys(proto_urls))
    weights_urls = list(dict.fromkeys(weights_urls))

    _download_with_fallback(proto, proto_urls, min_bytes=10_000)
    _download_with_fallback(weights, weights_urls, min_bytes=120_000_000)
    return proto, weights


def _detect_single_person(
    net: cv2.dnn_Net,
    frame: np.ndarray,
    input_size: int,
    kpt_thr: float,
) -> tuple[np.ndarray, np.ndarray]:
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0 / 255.0,
        size=(input_size, input_size),
        mean=(0, 0, 0),
        swapRB=False,
        crop=False,
    )
    net.setInput(blob)
    out = net.forward()
    n_parts = len(COCO_KEYPOINTS)
    heat_h, heat_w = out.shape[2], out.shape[3]

    points = np.zeros((n_parts, 2), dtype=np.float32)
    confs = np.zeros((n_parts,), dtype=np.float32)
    for i in range(n_parts):
        heat_map = out[0, i, :, :]
        _, conf, _, max_loc = cv2.minMaxLoc(heat_map)
        confs[i] = float(conf)
        if conf < kpt_thr:
            points[i, :] = -1.0
            continue
        x = (w * max_loc[0]) / heat_w
        y = (h * max_loc[1]) / heat_h
        points[i, :] = (x, y)
    return points, confs


def _draw_pose(
    frame: np.ndarray,
    points: np.ndarray,
    confs: np.ndarray,
    kpt_thr: float,
) -> None:
    for a, b in COCO_PAIRS:
        if confs[a] < kpt_thr or confs[b] < kpt_thr:
            continue
        xa, ya = int(points[a, 0]), int(points[a, 1])
        xb, yb = int(points[b, 0]), int(points[b, 1])
        if xa < 0 or ya < 0 or xb < 0 or yb < 0:
            continue
        cv2.line(frame, (xa, ya), (xb, yb), (0, 255, 255), 2, cv2.LINE_AA)

    for i in range(points.shape[0]):
        if confs[i] < kpt_thr:
            continue
        x, y = int(points[i, 0]), int(points[i, 1])
        if x < 0 or y < 0:
            continue
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OpenPose (OpenCV DNN COCO model) in real time."
    )
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--width", type=int, default=960, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=540, help="Requested camera height.")
    parser.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="OpenPose input size (256/320/368). Smaller is faster.",
    )
    parser.add_argument("--kpt-thr", type=float, default=0.12, help="Keypoint threshold.")
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.6,
        help="EMA smoothing for visible keypoints (0=no smoothing, 0.9=very smooth).",
    )
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="CPU thread cap for OpenCV and OpenMP.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / ".cache_openpose" / "models",
        help="Directory for OpenPose model files.",
    )
    parser.add_argument(
        "--proto-url",
        type=str,
        default=None,
        help="Optional override URL for OpenPose COCO prototxt.",
    )
    parser.add_argument(
        "--weights-url",
        type=str,
        default=None,
        help="Optional override URL for OpenPose COCO caffemodel.",
    )
    parser.add_argument("--show-fps", action="store_true", help="Show moving-average FPS.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.smooth_alpha = min(max(args.smooth_alpha, 0.0), 0.95)
    args.input_size = max(128, int(args.input_size))

    if args.cpu_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))
        cv2.setNumThreads(int(args.cpu_threads))

    try:
        proto_path, weights_path = _resolve_model_files(
            args.model_dir,
            args.proto_url,
            args.weights_url,
        )
    except Exception as exc:
        raise RuntimeError(
            "Failed to fetch OpenPose model files. Check internet access, pass --proto-url/--weights-url, "
            f"or place files manually in '{args.model_dir}'."
        ) from exc

    net = cv2.dnn.readNetFromCaffe(str(proto_path), str(weights_path))
    if cv2.ocl.haveOpenCL():
        cv2.ocl.setUseOpenCL(True)

    grabber = FrameGrabber(args.camera_id, args.width, args.height)
    grabber.start()

    window_name = "OpenPose Realtime (q/esc to quit)"
    fps_window = deque(maxlen=40)
    last_t = None
    last_seq = -1

    smooth_points: np.ndarray | None = None
    smooth_confs: np.ndarray | None = None

    try:
        while True:
            ok, frame, seq = grabber.read_latest()
            if not ok:
                time.sleep(0.002)
                continue
            if seq == last_seq:
                time.sleep(0.001)
                continue
            last_seq = seq

            points, confs = _detect_single_person(
                net=net,
                frame=frame,
                input_size=args.input_size,
                kpt_thr=args.kpt_thr,
            )

            if smooth_points is None or smooth_points.shape != points.shape:
                smooth_points = points.copy()
                smooth_confs = confs.copy()
            else:
                visible = confs >= args.kpt_thr
                if np.any(visible):
                    smooth_points[visible] = (
                        args.smooth_alpha * smooth_points[visible]
                        + (1.0 - args.smooth_alpha) * points[visible]
                    )
                    smooth_confs[visible] = confs[visible]
                hidden = ~visible
                if np.any(hidden):
                    smooth_confs[hidden] = np.maximum(0.0, smooth_confs[hidden] * 0.9)

            _draw_pose(frame, smooth_points, smooth_confs, args.kpt_thr)

            if args.show_fps:
                now = time.perf_counter()
                if last_t is not None:
                    fps_window.append(now - last_t)
                last_t = now
                fps = len(fps_window) / max(sum(fps_window), 1e-6) if fps_window else 0.0
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f} | OpenPose COCO | in:{args.input_size}",
                    (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        grabber.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
