"""Bounded, capture-time synchronization for an offline or local camera rig.

All timestamps use seconds on one local monotonic timeline after applying each
camera's configured offset: ``local_capture = camera_capture + clock_offset``.
Receive timestamps describe transport latency; they never become capture times.
The caller must obtain clock offsets independently (hardware synchronization,
PTP, or a measured clock calibration). This module does not estimate clocks.

Packets may arrive out of order within ``reorder_horizon_seconds`` of the latest
capture seen. Older packets are rejected. A bundle is finalized once that
watermark passes its complete skew window. A stopped camera therefore produces
partial bundles as other cameras advance; ``flush`` drains a finite recording.
Explicit disconnect/reconnect handles camera restarts without accepting delayed
packets from an earlier stream epoch. No background timers or wall clocks are
used, which makes recording replay and tests deterministic.
"""

from collections import Counter
from dataclasses import dataclass, field
from numbers import Integral, Real
from threading import RLock
import math
from uuid import uuid4

from .multicamera_types import CalibratedCamera, CameraFrame, FrameBundle, validated_identifier


class SynchronizationError(ValueError):
    """Invalid synchronizer configuration or lifecycle request."""


def _finite_number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise SynchronizationError(f"{name} must be a finite number")
    result = float(value)
    if result < 0 or (positive and result == 0):
        raise SynchronizationError(f"{name} must be {'positive' if positive else 'nonnegative'}")
    return result


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise SynchronizationError(f"{name} must be a positive integer")
    return int(value)


@dataclass
class _CameraState:
    offset: float
    uncertainty: float
    active: bool = True
    epoch: str | None = None
    retired_epochs: set = field(default_factory=set)
    highest_sequence: int = -1
    seen_sequences: dict = field(default_factory=dict)
    frames: list = field(default_factory=list)
    last_emitted_capture: float = float("-inf")


class FrameSynchronizer:
    """Thread-safe deterministic synchronizer with finite packet/replay buffers.

    ``max_skew_seconds`` bounds the entire bundle, including the worst-case
    camera clock uncertainty interval. A frame whose own uncertainty interval
    exceeds that budget is rejected, even if it would be a singleton. Output target time is the earliest
    corrected capture in that bundle. A frame is consumed at most once.

    Sequence numbers may reorder within a finite replay window. A jump of more
    than that window intentionally makes older sequence numbers ineligible.
    Stream epochs must be unique across reconnects; callers should generate a
    fresh UUID when a camera restarts and resets its sequence counter.
    """

    def __init__(
        self,
        camera_configs,
        *,
        max_skew_seconds=0.025,
        reorder_horizon_seconds=0.100,
        max_buffered_frames_per_camera=120,
        max_frame_age_seconds=5.0,
        max_future_lead_seconds=0.050,
        replay_window_size=4096,
        max_retired_stream_epochs=1024,
        session_id=None,
    ):
        cameras = tuple(camera_configs)
        if not cameras or any(not isinstance(camera, CalibratedCamera) for camera in cameras):
            raise SynchronizationError("camera_configs must contain calibrated cameras")
        if len({camera.camera_id for camera in cameras}) != len(cameras):
            raise SynchronizationError("camera IDs must be unique")
        self.max_skew_seconds = _finite_number(max_skew_seconds, "max_skew_seconds")
        self.reorder_horizon_seconds = _finite_number(reorder_horizon_seconds, "reorder_horizon_seconds")
        self.max_buffered_frames_per_camera = _positive_integer(
            max_buffered_frames_per_camera, "max_buffered_frames_per_camera"
        )
        self.max_frame_age_seconds = _finite_number(max_frame_age_seconds, "max_frame_age_seconds", positive=True)
        self.max_future_lead_seconds = _finite_number(max_future_lead_seconds, "max_future_lead_seconds")
        self.replay_window_size = _positive_integer(replay_window_size, "replay_window_size")
        self.max_retired_stream_epochs = _positive_integer(max_retired_stream_epochs, "max_retired_stream_epochs")
        self.session_id = uuid4().hex if session_id is None else validated_identifier(session_id, "session_id")
        if len(self.session_id) > 96:
            raise SynchronizationError("session_id must be at most 96 characters")
        self._cameras = {
            camera.camera_id: _CameraState(camera.clock_offset_seconds, camera.clock_uncertainty_seconds)
            for camera in sorted(cameras, key=lambda item: item.camera_id)
        }
        self._lock = RLock()
        self._counts = Counter()
        self._latest_capture = float("-inf")
        self._last_target = float("-inf")
        self._sealed_capture = float("-inf")
        self._next_bundle_id = 0
        self._last_advance_time = float("-inf")

    @property
    def stats(self):
        """Independent diagnostic snapshot; modifying it cannot change state."""
        with self._lock:
            result = dict(self._counts)
            for key in (
                "accepted_frames", "duplicate_frames", "stale_frames", "future_frames",
                "late_frames", "inactive_frames", "unknown_camera_frames",
                "epoch_mismatch_frames", "replay_window_frames", "dropped_buffer_frames",
                "disconnected_buffer_frames", "emitted_bundles", "emitted_frames", "partial_bundles",
                "invalid_timestamp_frames",
                "nonmonotonic_capture_frames",
                "uncertain_clock_frames",
            ):
                result.setdefault(key, 0)
            result["buffered_frames"] = sum(len(state.frames) for state in self._cameras.values())
            result["replay_entries"] = sum(len(state.seen_sequences) for state in self._cameras.values())
            result["retired_epochs"] = sum(len(state.retired_epochs) for state in self._cameras.values())
            return result

    def add_frame(self, frame):
        """Accept a packet and return newly finalized bundles, possibly empty.

        Malformed frame objects raise. Well-formed unusable packets are dropped
        with a reason in ``stats``. Rejected packets never advance the watermark.
        """
        if not isinstance(frame, CameraFrame):
            raise SynchronizationError("frame must be a CameraFrame")
        with self._lock:
            state = self._cameras.get(frame.camera_id)
            if state is None:
                self._counts["unknown_camera_frames"] += 1
                return ()
            if not state.active:
                self._counts["inactive_frames"] += 1
                return ()
            if frame.stream_epoch in state.retired_epochs or (
                state.epoch is not None and frame.stream_epoch != state.epoch
            ):
                self._counts["epoch_mismatch_frames"] += 1
                return ()
            # Even a singleton's possible capture interval must fit the same
            # window independently enforced by the downstream fusion boundary.
            # Leaving this check to pairwise matching would emit unusable
            # singleton bundles when a camera's clock calibration is too weak.
            if 2 * state.uncertainty > self.max_skew_seconds + 1e-12:
                self._counts["uncertain_clock_frames"] += 1
                return ()
            if frame.sequence in state.seen_sequences:
                self._counts["duplicate_frames"] += 1
                return ()
            if frame.sequence <= state.highest_sequence - self.replay_window_size:
                self._counts["replay_window_frames"] += 1
                return ()
            captured = frame.captured_at_seconds + state.offset
            if not math.isfinite(captured) or captured < 0:
                self._counts["invalid_timestamp_frames"] += 1
                return ()
            if captured - frame.received_at_seconds > self.max_future_lead_seconds + 1e-12:
                self._counts["future_frames"] += 1
                return ()
            if frame.received_at_seconds - captured > self.max_frame_age_seconds + 1e-12:
                self._counts["stale_frames"] += 1
                return ()
            # A restarted capture clock requires explicit new epoch/offset.
            # Network reordering changes delivery order, not the capture order
            # encoded by monotonically increasing source sequence numbers.
            if any((sequence < frame.sequence and timestamp > captured)
                   or (sequence > frame.sequence and timestamp < captured)
                   for sequence, timestamp in state.seen_sequences.items()):
                self._counts["nonmonotonic_capture_frames"] += 1
                return ()
            watermark = self._latest_capture - self.reorder_horizon_seconds
            if (captured < watermark - 1e-12 or captured <= self._sealed_capture
                    or captured <= self._last_target or captured <= state.last_emitted_capture):
                self._counts["late_frames"] += 1
                return ()
            if any(abs(captured - item[0]) <= 1e-12 for item in state.frames):
                self._counts["duplicate_frames"] += 1
                return ()
            state.epoch = frame.stream_epoch
            state.highest_sequence = max(frame.sequence, state.highest_sequence)
            state.seen_sequences[frame.sequence] = captured
            floor = state.highest_sequence - self.replay_window_size
            state.seen_sequences = {sequence: timestamp for sequence, timestamp in state.seen_sequences.items() if sequence > floor}
            state.frames.append((captured, frame))
            state.frames.sort(key=lambda item: (item[0], item[1].sequence))
            self._latest_capture = max(captured, self._latest_capture)
            self._counts["accepted_frames"] += 1
            # Finalize safe windows before applying the hard memory cap.
            bundles = self._drain(flush=False)
            if len(state.frames) > self.max_buffered_frames_per_camera:
                overflow = len(state.frames) - self.max_buffered_frames_per_camera
                del state.frames[:overflow]
                self._counts["dropped_buffer_frames"] += overflow
            return bundles

    def flush(self):
        """Finalize all buffered frames; use at end-of-recording or explicit cut."""
        with self._lock:
            bundles = self._drain(flush=True)
            self._sealed_capture = max(self._sealed_capture, self._latest_capture)
            return bundles

    def advance_watermark(self, now_seconds):
        """Finalize waited partials on silence using the caller's local clock.

        Supply the same monotonic timeline used by corrected capture times.
        This is an explicit local timer event, never a packet's receive time.
        Later packets outside the configured lateness budget are rejected.
        """
        now = _finite_number(now_seconds, "now_seconds")
        with self._lock:
            if now < self._last_advance_time:
                raise SynchronizationError("watermark time cannot move backwards")
            self._last_advance_time = now
            self._latest_capture = max(self._latest_capture, now)
            return self._drain(flush=False)

    def disconnect_camera(self, camera_id):
        """Discard the camera's pending packets; other views remain available."""
        with self._lock:
            state = self._camera(camera_id)
            state.active = False
            self._counts["disconnected_buffer_frames"] += len(state.frames)
            state.frames.clear()
            return self._drain(flush=False)

    def connect_camera(self, camera_id, *, stream_epoch=None, clock_offset_seconds=None,
                       clock_uncertainty_seconds=None):
        """Resume a stream, explicitly rotating epoch when its sequence resets.

        Changing clock calibration while packets are buffered would reinterpret
        old data, so those packets are discarded. Past bundles are immutable.
        Previously retired epochs cannot be reactivated.
        """
        with self._lock:
            state = self._camera(camera_id)
            if stream_epoch is not None:
                validated_identifier(stream_epoch, "stream_epoch")
            if stream_epoch in state.retired_epochs:
                raise SynchronizationError("a retired stream epoch cannot be reactivated")
            if clock_offset_seconds is not None:
                # Offsets, unlike durations, are allowed to be negative.
                if isinstance(clock_offset_seconds, bool) or not isinstance(clock_offset_seconds, Real) or not math.isfinite(clock_offset_seconds):
                    raise SynchronizationError("clock_offset_seconds must be finite")
            uncertainty = state.uncertainty if clock_uncertainty_seconds is None else _finite_number(
                clock_uncertainty_seconds, "clock_uncertainty_seconds"
            )
            epoch_changed = stream_epoch is not None and stream_epoch != state.epoch
            calibration_changed = clock_offset_seconds is not None or clock_uncertainty_seconds is not None
            if epoch_changed and state.epoch is not None and len(state.retired_epochs) >= self.max_retired_stream_epochs:
                raise SynchronizationError("retired epoch limit reached; start a new synchronizer session")
            if epoch_changed or calibration_changed:
                self._counts["disconnected_buffer_frames"] += len(state.frames)
                state.frames.clear()
            if epoch_changed:
                if state.epoch is not None:
                    state.retired_epochs.add(state.epoch)
                state.epoch = stream_epoch
                state.highest_sequence = -1
                state.seen_sequences.clear()
                state.last_emitted_capture = float("-inf")
            if clock_offset_seconds is not None:
                state.offset = float(clock_offset_seconds)
            state.uncertainty = uncertainty
            state.active = True

    def _camera(self, camera_id):
        try:
            return self._cameras[camera_id]
        except (KeyError, TypeError) as exc:
            raise SynchronizationError("unknown camera ID") from exc

    def _drain(self, *, flush):
        bundles = []
        watermark = self._latest_capture - self.reorder_horizon_seconds
        while True:
            available = [(state.frames[0][0], camera_id) for camera_id, state in self._cameras.items() if state.frames]
            if not available:
                break
            anchor, anchor_id = min(available)
            if not flush and anchor + self.max_skew_seconds > watermark + 1e-12:
                break
            chosen = []
            chosen_times = []
            chosen_uncertainties = []
            # Include anchor first; subsequent candidates must meet every
            # selected view's pairwise worst-case clock uncertainty bound.
            camera_order = (anchor_id, *(key for key in self._cameras if key != anchor_id))
            for camera_id in camera_order:
                state = self._cameras[camera_id]
                if not state.frames:
                    continue
                captured, frame = state.frames[0]
                if captured - anchor > self.max_skew_seconds + 1e-12:
                    continue
                if any(abs(captured - old) + state.uncertainty + uncertainty > self.max_skew_seconds + 1e-12
                       for old, uncertainty in zip(chosen_times, chosen_uncertainties)):
                    continue
                state.frames.pop(0)
                state.last_emitted_capture = captured
                chosen.append(frame)
                chosen_times.append(captured)
                chosen_uncertainties.append(state.uncertainty)
            chosen.sort(key=lambda frame: frame.camera_id)
            camera_ids = {frame.camera_id for frame in chosen}
            missing = tuple(key for key in self._cameras if key not in camera_ids)
            bundle = FrameBundle(
                bundle_id=f"{self.session_id}:{self._next_bundle_id}",
                target_timestamp_seconds=anchor,
                frames=tuple(chosen),
                missing_cameras=missing,
                span_seconds=max(chosen_times) - min(chosen_times),
            )
            self._next_bundle_id += 1
            self._last_target = anchor
            self._counts["emitted_bundles"] += 1
            self._counts["emitted_frames"] += len(chosen)
            if missing:
                self._counts["partial_bundles"] += 1
            bundles.append(bundle)
        return tuple(bundles)
