#!/usr/bin/env python3
"""Real-time YOLOv8n pose demo with tracker, smoothing, occlusion fill, and pseudo-3D."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import os
from pathlib import Path
import threading
import time

import cv2
import numpy as np

# Keep Ultralytics settings/cache writable inside project workspace.
_CACHE_ROOT = Path(__file__).resolve().parent / ".cache_yolo11"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_CACHE_ROOT))
_WEIGHTS_DIR = _CACHE_ROOT / "weights"
_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO, settings  # noqa: E402

settings.update(
    {
        "weights_dir": str(_WEIGHTS_DIR),
        "runs_dir": str(_CACHE_ROOT / "runs"),
    }
)


# COCO-17 skeleton edges for pose visualization.
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

# Leg joints in COCO-17.
LEG_KPTS = {11, 12, 13, 14, 15, 16}

MIRROR_KPT = {
    1: 2,
    2: 1,
    3: 4,
    4: 3,
    5: 6,
    6: 5,
    7: 8,
    8: 7,
    9: 10,
    10: 9,
    11: 12,
    12: 11,
    13: 14,
    14: 13,
    15: 16,
    16: 15,
}


@dataclass
class TrackState:
    xy: np.ndarray
    vel: np.ndarray
    conf: np.ndarray
    missing: np.ndarray
    z: np.ndarray
    torso_ref: float
    last_ts: float


class FrameGrabber:
    """Background frame reader that always keeps only the latest frame."""

    def __init__(self, source: int, width: int, height: int) -> None:
        self._source = source
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = False
        self._latest_frame: np.ndarray | None = None
        self._latest_seq = -1

    def start(self) -> None:
        cap = cv2.VideoCapture(self._source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera index {self._source}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def _reader_loop(self) -> None:
        assert self._cap is not None
        seq = 0
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._latest_frame = frame
                self._latest_seq = seq
            seq += 1

    def read_latest(self) -> tuple[bool, np.ndarray | None, int]:
        with self._lock:
            if self._latest_frame is None:
                return False, None, self._latest_seq
            return True, self._latest_frame.copy(), self._latest_seq

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device

    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "0"
    except Exception:
        pass
    return "cpu"


def _torso_metrics(
    xy: np.ndarray, conf: np.ndarray, conf_thr: float
) -> tuple[float, np.ndarray, float]:
    # Uses shoulders/hips when visible to get a stable body scale and center.
    idx = [5, 6, 11, 12]
    visible = [i for i in idx if conf[i] >= conf_thr]
    if len(visible) < 2:
        center = np.array([np.mean(xy[:, 0]), np.mean(xy[:, 1])], dtype=np.float32)
        return 0.0, center, 0.0

    center = np.mean(xy[visible], axis=0).astype(np.float32)
    shoulder_vis = conf[5] >= conf_thr and conf[6] >= conf_thr
    hip_vis = conf[11] >= conf_thr and conf[12] >= conf_thr

    shoulder_w = float(np.linalg.norm(xy[6] - xy[5])) if shoulder_vis else 0.0
    hip_w = float(np.linalg.norm(xy[12] - xy[11])) if hip_vis else shoulder_w

    if shoulder_vis and hip_vis:
        torso_len = float(np.linalg.norm((xy[5] + xy[6]) * 0.5 - (xy[11] + xy[12]) * 0.5))
    elif shoulder_vis:
        torso_len = shoulder_w * 1.6
    elif hip_vis:
        torso_len = hip_w * 1.6
    else:
        torso_len = 0.0

    return torso_len, center, shoulder_w if shoulder_w > 0.0 else hip_w


def _attempt_mirror_fill(
    joint_idx: int,
    xy: np.ndarray,
    conf: np.ndarray,
    conf_thr: float,
) -> np.ndarray | None:
    if joint_idx not in MIRROR_KPT:
        return None

    mirror_idx = MIRROR_KPT[joint_idx]
    if conf[mirror_idx] < conf_thr:
        return None

    anchors = [i for i in (5, 6, 11, 12) if conf[i] >= conf_thr]
    if not anchors:
        return None

    axis_x = float(np.mean(xy[anchors, 0]))
    est = xy[mirror_idx].copy()
    est[0] = 2.0 * axis_x - est[0]
    return est


def _draw_pose(
    frame: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    inferred: np.ndarray,
    kpt_thr: float,
    edge_margin: int,
) -> None:
    h, w = frame.shape[:2]

    def _inside_safe(pt: np.ndarray) -> bool:
        x, y = float(pt[0]), float(pt[1])
        return edge_margin <= x < (w - edge_margin) and edge_margin <= y < (h - edge_margin)

    for a, b in COCO17_EDGES:
        if keypoints_conf[a] < kpt_thr or keypoints_conf[b] < kpt_thr:
            continue
        # Do not draw inferred links that touch unsafe border areas.
        if (inferred[a] and not _inside_safe(keypoints_xy[a])) or (
            inferred[b] and not _inside_safe(keypoints_xy[b])
        ):
            continue
        xa, ya = int(keypoints_xy[a, 0]), int(keypoints_xy[a, 1])
        xb, yb = int(keypoints_xy[b, 0]), int(keypoints_xy[b, 1])
        color = (0, 180, 255) if inferred[a] or inferred[b] else (0, 255, 255)
        cv2.line(frame, (xa, ya), (xb, yb), color, 2, cv2.LINE_AA)

    for i in range(keypoints_xy.shape[0]):
        if keypoints_conf[i] < kpt_thr:
            continue
        if inferred[i] and not _inside_safe(keypoints_xy[i]):
            continue
        x, y = int(keypoints_xy[i, 0]), int(keypoints_xy[i, 1])
        if inferred[i]:
            cv2.circle(frame, (x, y), 4, (0, 180, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1, cv2.LINE_AA)


def _build_pseudo_3d(
    xy: np.ndarray,
    conf: np.ndarray,
    conf_thr: float,
    torso_ref: float,
    depth_scale: float,
    depth_lr_scale: float,
    depth_vertical_scale: float,
) -> tuple[np.ndarray, float]:
    torso_len, center, torso_w = _torso_metrics(xy, conf, conf_thr)
    if torso_ref <= 1e-4 and torso_len > 1e-4:
        torso_ref = torso_len
    if torso_ref <= 1e-4:
        torso_ref = 120.0
    if torso_len <= 1e-4:
        torso_len = torso_ref
    if torso_w <= 1e-4:
        torso_w = torso_len * 0.8

    relative_depth = (torso_ref / max(torso_len, 1e-4)) - 1.0
    base_z = depth_scale * relative_depth * torso_ref

    xyz = np.zeros((xy.shape[0], 3), dtype=np.float32)
    xyz[:, :2] = xy
    lr = (xy[:, 0] - center[0]) / max(torso_w, 1e-4)
    vertical = (center[1] - xy[:, 1]) / max(torso_len, 1e-4)
    xyz[:, 2] = (
        base_z
        + (depth_lr_scale * lr * torso_ref)
        + (depth_vertical_scale * vertical * torso_ref)
    )
    return xyz, torso_len


def _draw_3d_panel(
    panel: np.ndarray,
    xyz: np.ndarray,
    conf: np.ndarray,
    inferred: np.ndarray,
    kpt_thr: float,
    color: tuple[int, int, int],
) -> None:
    h, w = panel.shape[:2]
    center = np.array([w * 0.5, h * 0.62], dtype=np.float32)
    pts = xyz.copy()
    body_center = np.mean(pts[[5, 6, 11, 12]], axis=0)
    pts -= body_center

    # Slight y-rotation to make depth changes visible.
    yaw = np.deg2rad(28.0)
    cos_y, sin_y = float(np.cos(yaw)), float(np.sin(yaw))
    rot = np.array([[cos_y, 0.0, sin_y], [0.0, 1.0, 0.0], [-sin_y, 0.0, cos_y]], dtype=np.float32)
    pts = pts @ rot.T

    torso = np.linalg.norm(xyz[6, :2] - xyz[5, :2]) + np.linalg.norm(xyz[12, :2] - xyz[11, :2])
    scale = max(0.35, min(1.15, (h * 0.55) / max(torso, 60.0)))

    proj = np.zeros((pts.shape[0], 2), dtype=np.float32)
    proj[:, 0] = center[0] + pts[:, 0] * scale
    proj[:, 1] = center[1] + pts[:, 1] * scale - pts[:, 2] * scale * 0.55

    for a, b in COCO17_EDGES:
        if conf[a] < kpt_thr or conf[b] < kpt_thr:
            continue
        xa, ya = int(proj[a, 0]), int(proj[a, 1])
        xb, yb = int(proj[b, 0]), int(proj[b, 1])
        c = (0, 180, 255) if inferred[a] or inferred[b] else color
        cv2.line(panel, (xa, ya), (xb, yb), c, 2, cv2.LINE_AA)

    for i in range(proj.shape[0]):
        if conf[i] < kpt_thr:
            continue
        x, y = int(proj[i, 0]), int(proj[i, 1])
        c = (0, 180, 255) if inferred[i] else color
        cv2.circle(panel, (x, y), 3, c, -1, cv2.LINE_AA)


def _track_color(track_id: int) -> tuple[int, int, int]:
    seed = int(track_id) * 2654435761
    b = 40 + (seed & 0x7F)
    g = 100 + ((seed >> 8) & 0x7F)
    r = 120 + ((seed >> 16) & 0x7F)
    return int(b), int(g), int(r)


def _clip_point(pt: np.ndarray, w: int, h: int) -> tuple[np.ndarray, bool]:
    x = float(np.clip(pt[0], 0.0, max(0.0, float(w - 1))))
    y = float(np.clip(pt[1], 0.0, max(0.0, float(h - 1))))
    clipped = abs(x - float(pt[0])) > 1e-4 or abs(y - float(pt[1])) > 1e-4
    return np.array([x, y], dtype=np.float32), clipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8n-pose in real time with tracker, smoothing, and pseudo-3D."
    )
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-pose.pt",
        help="Ultralytics pose model. Example: yolov8n-pose.pt / yolov8s-pose.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "mps"),
        help="Inference device.",
    )
    parser.add_argument("--imgsz", type=int, default=512, help="Inference size.")
    parser.add_argument("--conf", type=float, default=0.4, help="Detection confidence.")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold.")
    parser.add_argument("--kpt-thr", type=float, default=0.3, help="Keypoint draw threshold.")
    parser.add_argument(
        "--smooth-alpha",
        type=float,
        default=0.65,
        help="EMA smoothing for visible keypoints (0=no smoothing, 0.9=very smooth).",
    )
    parser.add_argument(
        "--velocity-alpha",
        type=float,
        default=0.7,
        help="Smoothing factor for landmark velocity used in occlusion prediction.",
    )
    parser.add_argument(
        "--max-occlusion-frames",
        type=int,
        default=10,
        help="Frames to keep predicting hidden landmarks per track.",
    )
    parser.add_argument(
        "--occlusion-decay",
        type=float,
        default=0.82,
        help="Velocity decay while a landmark stays occluded.",
    )
    parser.add_argument(
        "--inferred-conf",
        type=float,
        default=0.36,
        help="Confidence assigned to predicted hidden landmarks.",
    )
    parser.add_argument("--max-det", type=int, default=4, help="Maximum people per frame.")
    parser.add_argument("--width", type=int, default=960, help="Requested camera width.")
    parser.add_argument("--height", type=int, default=540, help="Requested camera height.")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=4,
        help="CPU thread cap when running on CPU.",
    )
    parser.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Ultralytics tracker config file.",
    )
    parser.add_argument(
        "--edge-margin",
        type=int,
        default=10,
        help="Suppress inferred keypoints/links near image borders (pixels).",
    )
    parser.add_argument(
        "--show-3d",
        action="store_true",
        help="Show pseudo-3D skeleton panel beside the 2D view.",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=0.85,
        help="Global pseudo depth strength.",
    )
    parser.add_argument(
        "--depth-lr-scale",
        type=float,
        default=0.18,
        help="Left-right pseudo depth strength.",
    )
    parser.add_argument(
        "--depth-vertical-scale",
        type=float,
        default=0.08,
        help="Vertical pseudo depth strength.",
    )
    parser.add_argument("--show-fps", action="store_true", help="Show moving-average FPS.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.smooth_alpha = min(max(args.smooth_alpha, 0.0), 0.97)
    args.velocity_alpha = min(max(args.velocity_alpha, 0.0), 0.98)
    args.occlusion_decay = min(max(args.occlusion_decay, 0.0), 1.0)
    args.edge_margin = max(int(args.edge_margin), 0)

    device = _resolve_device(args.device)
    if device == "cpu" and args.cpu_threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
        os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
        os.environ.setdefault("MKL_NUM_THREADS", str(args.cpu_threads))

    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute() and model_path.parent == Path("."):
        model_path = _WEIGHTS_DIR / model_path.name
    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        msg = str(exc)
        if "Download failure" in msg or "Environment may be offline" in msg:
            raise RuntimeError(
                f"Could not download model '{args.model}'. Place the weights file at "
                f"'{model_path}' and run again, or run once with internet access."
            ) from exc
        raise

    grabber = FrameGrabber(args.camera_id, args.width, args.height)
    grabber.start()

    window_name = "YOLOv8n Pose + Tracker + Pseudo-3D (q/esc to quit)"
    fps_window = deque(maxlen=50)
    last_t = None
    last_seq = -1

    track_state: dict[int, TrackState] = {}
    no_id_counter = 0

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
            now = time.perf_counter()

            results = model.track(
                source=frame,
                persist=True,
                tracker=args.tracker,
                device=device,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                max_det=args.max_det,
                verbose=False,
            )
            result = results[0]
            panel_3d = None
            if args.show_3d:
                panel_3d = np.zeros((frame.shape[0], 360, 3), dtype=np.uint8)
                cv2.putText(
                    panel_3d,
                    "Pseudo-3D (relative depth)",
                    (16, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (210, 210, 210),
                    2,
                    cv2.LINE_AA,
                )

            if result.keypoints is not None and len(result.keypoints) > 0:
                xy = result.keypoints.xy.cpu().numpy().astype(np.float32)  # (N, K, 2)
                conf = result.keypoints.conf
                conf_np = conf.cpu().numpy().astype(np.float32) if conf is not None else None

                boxes = result.boxes
                if boxes is not None and boxes.id is not None:
                    track_ids = boxes.id.int().cpu().tolist()
                else:
                    track_ids = []
                    for _ in range(xy.shape[0]):
                        no_id_counter += 1
                        track_ids.append(10_000_000 + no_id_counter)

                live_ids = set(track_ids)
                for stale_id in list(track_state.keys()):
                    if stale_id not in live_ids:
                        del track_state[stale_id]

                for i, tid in enumerate(track_ids):
                    obs_xy = xy[i]
                    frame_h, frame_w = frame.shape[:2]
                    obs_xy[:, 0] = np.clip(obs_xy[:, 0], 0.0, max(0.0, float(frame_w - 1)))
                    obs_xy[:, 1] = np.clip(obs_xy[:, 1], 0.0, max(0.0, float(frame_h - 1)))
                    obs_conf = (
                        np.ones(obs_xy.shape[0], dtype=np.float32)
                        if conf_np is None
                        else conf_np[i].copy()
                    )
                    prev = track_state.get(tid)
                    if prev is None:
                        torso_len, _, _ = _torso_metrics(obs_xy, obs_conf, args.kpt_thr)
                        torso_ref = torso_len if torso_len > 1e-4 else 120.0
                        prev = TrackState(
                            xy=obs_xy.copy(),
                            vel=np.zeros_like(obs_xy, dtype=np.float32),
                            conf=obs_conf.copy(),
                            missing=np.zeros(obs_xy.shape[0], dtype=np.int16),
                            z=np.zeros(obs_xy.shape[0], dtype=np.float32),
                            torso_ref=float(torso_ref),
                            last_ts=now,
                        )
                        track_state[tid] = prev

                    dt = max(now - prev.last_ts, 1.0 / 180.0)
                    out_xy = obs_xy.copy()
                    out_conf = obs_conf.copy()
                    inferred = np.zeros(obs_xy.shape[0], dtype=bool)

                    # Pass 1: smooth visible joints and update velocity.
                    visible = obs_conf >= args.kpt_thr
                    if np.any(visible):
                        delta = (obs_xy[visible] - prev.xy[visible]) / dt
                        # Cap visible-joint velocity to prevent large spikes near borders.
                        speed_cap = max(prev.torso_ref * 4.0, 220.0)
                        delta_norm = np.linalg.norm(delta, axis=1, keepdims=True)
                        delta_scale = np.minimum(1.0, speed_cap / np.maximum(delta_norm, 1e-6))
                        delta = delta * delta_scale
                        prev.vel[visible] = (
                            args.velocity_alpha * prev.vel[visible]
                            + (1.0 - args.velocity_alpha) * delta
                        )
                        out_xy[visible] = (
                            args.smooth_alpha * prev.xy[visible]
                            + (1.0 - args.smooth_alpha) * obs_xy[visible]
                        )
                        prev.missing[visible] = 0

                    # Pass 2: predict/fill occluded joints.
                    occluded_idx = np.where(~visible)[0]
                    for j in occluded_idx:
                        pred = None
                        max_step = max(prev.torso_ref * 0.18, 20.0)
                        if prev.missing[j] < args.max_occlusion_frames:
                            step = prev.vel[j] * dt
                            step_norm = float(np.linalg.norm(step))
                            if step_norm > max_step:
                                step = step * (max_step / max(step_norm, 1e-6))
                            pred = prev.xy[j] + step
                        if pred is None:
                            pred = _attempt_mirror_fill(j, out_xy, out_conf, args.kpt_thr)
                            if pred is not None:
                                # Reject mirror fill if it jumps too far.
                                jump = float(np.linalg.norm(pred - prev.xy[j]))
                                if jump > max(prev.torso_ref * 0.65, 80.0):
                                    pred = None
                        if pred is None:
                            pred = prev.xy[j]

                        pred, clipped = _clip_point(pred, frame_w, frame_h)
                        out_xy[j] = pred

                        # Fade inferred confidence as occlusion length grows.
                        remain = 1.0 - min(
                            1.0, float(prev.missing[j]) / max(1.0, float(args.max_occlusion_frames))
                        )
                        inferred_conf = max(0.05, float(args.inferred_conf) * remain)
                        out_conf[j] = max(float(out_conf[j]), inferred_conf)

                        # For inferred leg points clipped on border, avoid drawing unstable links.
                        if clipped and j in LEG_KPTS:
                            out_conf[j] = min(out_conf[j], args.kpt_thr - 1e-3)

                        prev.vel[j] *= args.occlusion_decay
                        prev.missing[j] += 1
                        inferred[j] = True

                    # Keep final joints inside frame bounds.
                    out_xy[:, 0] = np.clip(out_xy[:, 0], 0.0, max(0.0, float(frame_w - 1)))
                    out_xy[:, 1] = np.clip(out_xy[:, 1], 0.0, max(0.0, float(frame_h - 1)))

                    xyz, torso_len = _build_pseudo_3d(
                        out_xy,
                        out_conf,
                        args.kpt_thr,
                        prev.torso_ref,
                        args.depth_scale,
                        args.depth_lr_scale,
                        args.depth_vertical_scale,
                    )
                    if torso_len > 1e-4:
                        prev.torso_ref = 0.95 * prev.torso_ref + 0.05 * torso_len
                    prev.z = xyz[:, 2].copy()
                    prev.xy = out_xy.copy()
                    prev.conf = out_conf.copy()
                    prev.last_ts = now

                    _draw_pose(
                        frame,
                        out_xy,
                        out_conf,
                        inferred,
                        args.kpt_thr,
                        args.edge_margin,
                    )
                    if panel_3d is not None:
                        _draw_3d_panel(
                            panel_3d,
                            xyz,
                            out_conf,
                            inferred,
                            args.kpt_thr,
                            _track_color(int(tid)),
                        )
                    cv2.putText(
                        frame,
                        f"id:{tid}",
                        (int(out_xy[5, 0]), int(out_xy[5, 1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            if args.show_fps:
                if last_t is not None:
                    fps_window.append(now - last_t)
                last_t = now
                fps = len(fps_window) / max(sum(fps_window), 1e-6) if fps_window else 0.0
                cv2.putText(
                    frame,
                    f"FPS:{fps:5.1f}  model:{Path(str(model_path)).name}  dev:{device}",
                    (14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.64,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            out_frame = np.hstack([frame, panel_3d]) if panel_3d is not None else frame
            cv2.imshow(window_name, out_frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        grabber.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
