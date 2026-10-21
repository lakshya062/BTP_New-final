"""Capture-clock synchronization, packet faults, reconnection, and load tests."""

from concurrent.futures import ThreadPoolExecutor
import random
import unittest

import numpy as np

from core.camera_calibration import CameraCalibration
from core.multicamera_fusion import MultiCameraFusion
from core.multicamera_sync import FrameSynchronizer, SynchronizationError
from core.multicamera_types import CalibratedCamera, CameraFrame


def camera(camera_id, offset=0.0, uncertainty=0.0):
    calibration = CameraCalibration(
        np.array([[800., 0., 320.], [0., 800., 240.], [0., 0., 1.]]),
        np.zeros(5), (640, 480),
    )
    return CalibratedCamera(camera_id, calibration, np.eye(3), np.zeros(3), offset, uncertainty)


def packet(camera_id, sequence=0, captured=10.0, received=None, epoch="default"):
    return CameraFrame(camera_id, sequence, captured, captured + 0.010 if received is None else received, (), epoch)


class FrameSynchronizerTest(unittest.TestCase):
    def sync(self, cameras=None, **kwargs):
        kwargs.setdefault("session_id", "test")
        return FrameSynchronizer(cameras or [camera("a"), camera("b")], **kwargs)

    def test_matching_captures_wait_for_watermark_then_emit_one_bundle(self):
        sync = self.sync()
        self.assertEqual(sync.add_frame(packet("a")), ())
        self.assertEqual(sync.add_frame(packet("b", captured=10.008)), ())
        bundles = sync.advance_watermark(10.2)
        self.assertEqual(len(bundles), 1)
        self.assertEqual(bundles[0].bundle_id, "test:0")
        self.assertEqual(tuple(frame.camera_id for frame in bundles[0].frames), ("a", "b"))
        self.assertEqual(bundles[0].target_timestamp_seconds, 10.0)
        self.assertAlmostEqual(bundles[0].span_seconds, 0.008)
        self.assertEqual(bundles[0].missing_cameras, ())

    def test_capture_offsets_correct_clocks_before_matching(self):
        sync = self.sync([camera("a", 0.5), camera("b", -0.7)])
        sync.add_frame(packet("a", captured=9.5, received=10.01))
        sync.add_frame(packet("b", captured=10.7, received=10.07))
        bundle, = sync.flush()
        self.assertAlmostEqual(bundle.target_timestamp_seconds, 10)
        self.assertAlmostEqual(bundle.span_seconds, 0)
        self.assertEqual(len(bundle.frames), 2)

    def test_receive_jitter_does_not_change_capture_alignment(self):
        sync = self.sync()
        sync.add_frame(packet("a", captured=10, received=10.9))
        sync.add_frame(packet("b", captured=10.001, received=10.002))
        bundle, = sync.flush()
        self.assertAlmostEqual(bundle.span_seconds, 0.001)
        self.assertEqual(len(bundle.frames), 2)

    def test_identical_arrival_times_cannot_merge_different_captures(self):
        sync = self.sync()
        sync.add_frame(packet("a", captured=10, received=10.1))
        sync.add_frame(packet("b", captured=10.08, received=10.1))
        bundles = sync.flush()
        self.assertEqual(len(bundles), 2)
        self.assertTrue(all(len(bundle.frames) == 1 for bundle in bundles))

    def test_reordering_within_horizon_is_sorted_before_bundling(self):
        sync = self.sync(reorder_horizon_seconds=0.3)
        for frame in (packet("a", 1, 10.1), packet("b", 0, 10), packet("a", 0, 10), packet("b", 1, 10.1)):
            sync.add_frame(frame)
        bundles = sync.flush()
        self.assertEqual([bundle.target_timestamp_seconds for bundle in bundles], [10, 10.1])
        self.assertEqual([[frame.sequence for frame in bundle.frames] for bundle in bundles], [[0, 0], [1, 1]])

    def test_beyond_horizon_packets_are_dropped_without_moving_frontier(self):
        sync = self.sync()
        sync.add_frame(packet("a", 2, 10.3))
        self.assertEqual(sync.add_frame(packet("b", 0, 10.1, received=10.4)), ())
        self.assertEqual(sync.stats["late_frames"], 1)
        self.assertEqual(sync.stats["accepted_frames"], 1)

    def test_exact_lateness_boundary_is_accepted(self):
        sync = self.sync(reorder_horizon_seconds=0.1)
        sync.add_frame(packet("a", 1, 10.1))
        sync.add_frame(packet("b", 0, 10.0))
        self.assertEqual(sync.stats["late_frames"], 0)

    def test_duplicate_packet_is_never_consumed_twice(self):
        sync = self.sync()
        frame = packet("a")
        sync.add_frame(frame)
        sync.add_frame(frame)
        self.assertEqual(sync.stats["duplicate_frames"], 1)
        self.assertEqual(sum(len(bundle.frames) for bundle in sync.flush()), 1)
        sync.add_frame(frame)
        self.assertEqual(sync.stats["duplicate_frames"], 2)

    def test_duplicate_capture_time_with_new_sequence_is_dropped(self):
        sync = self.sync()
        sync.add_frame(packet("a", 0))
        sync.add_frame(packet("a", 1))
        self.assertEqual(sync.stats["duplicate_frames"], 1)
        self.assertEqual(len(sync.flush()), 1)

    def test_missing_camera_produces_partial_bundle_and_recovers(self):
        sync = self.sync()
        sync.add_frame(packet("a", 0, 10))
        first, = sync.add_frame(packet("a", 1, 10.2))
        self.assertEqual(first.missing_cameras, ("b",))
        sync.add_frame(packet("b", 1, 10.201))
        second, = sync.flush()
        self.assertEqual(len(second.frames), 2)
        self.assertEqual(sync.stats["partial_bundles"], 1)

    def test_all_cameras_silent_timer_matures_pending_frame(self):
        sync = self.sync()
        sync.add_frame(packet("a"))
        self.assertEqual(sync.advance_watermark(10.05), ())
        bundle, = sync.advance_watermark(10.2)
        self.assertEqual(bundle.missing_cameras, ("b",))
        self.assertEqual(sync.stats["buffered_frames"], 0)
        self.assertEqual(sync.advance_watermark(10.3), ())

    def test_timer_cannot_move_backwards(self):
        sync = self.sync()
        sync.advance_watermark(10)
        with self.assertRaises(SynchronizationError):
            sync.advance_watermark(9)

    def test_timer_does_not_accept_late_packet_as_current(self):
        sync = self.sync()
        sync.advance_watermark(11)
        sync.add_frame(packet("a", captured=10))
        self.assertEqual(sync.stats["late_frames"], 1)

    def test_stale_packet_is_dropped(self):
        sync = self.sync(max_frame_age_seconds=0.2)
        sync.add_frame(packet("a", captured=10, received=10.3))
        self.assertEqual(sync.stats["stale_frames"], 1)
        self.assertEqual(sync.flush(), ())

    def test_future_packet_is_dropped_and_does_not_poison_watermark(self):
        sync = self.sync()
        sync.add_frame(packet("a", 999, 100, received=10))
        sync.add_frame(packet("a", 0, 10))
        self.assertEqual(sync.stats["future_frames"], 1)
        self.assertEqual(sync.stats["accepted_frames"], 1)
        self.assertEqual(len(sync.flush()), 1)

    def test_clock_correction_cannot_produce_negative_or_infinite_capture(self):
        sync = self.sync([camera("a", -20)])
        sync.add_frame(packet("a", captured=10))
        self.assertEqual(sync.stats["invalid_timestamp_frames"], 1)
        sync = self.sync([camera("a", 1e308)])
        sync.add_frame(packet("a", captured=1e308))
        self.assertEqual(sync.stats["invalid_timestamp_frames"], 1)

    def test_uncertainty_is_pairwise_and_can_force_partial_bundles(self):
        sync = self.sync([camera("a", uncertainty=0.010), camera("b", uncertainty=0.010)])
        sync.add_frame(packet("a"))
        sync.add_frame(packet("b", captured=10.008))
        bundles = sync.flush()
        self.assertEqual(len(bundles), 2)
        self.assertTrue(all(len(bundle.frames) == 1 for bundle in bundles))

    def test_uncertain_camera_is_dropped_while_healthy_views_reach_real_fusion(self):
        cameras = [camera("a", uncertainty=.020), camera("b", uncertainty=.002), camera("c", uncertainty=.002)]
        sync = self.sync(cameras, max_skew_seconds=.025)
        fusion = MultiCameraFusion(cameras, max_span_seconds=.025)
        for name in "abc":
            sync.add_frame(packet(name))
        bundle, = sync.flush()
        self.assertEqual(tuple(frame.camera_id for frame in bundle.frames), ("b", "c"))
        self.assertEqual(bundle.missing_cameras, ("a",))
        self.assertEqual(sync.stats["uncertain_clock_frames"], 1)
        result = fusion.fuse(bundle)
        self.assertEqual(result.missing_cameras, ("a",))
        self.assertEqual(result.persons, ())

    def test_uncertain_single_camera_never_emits_invalid_fusion_bundle_and_recovers(self):
        sync = self.sync([camera("a", uncertainty=.020)], max_skew_seconds=.025)
        self.assertEqual(sync.add_frame(packet("a")), ())
        self.assertEqual(sync.flush(), ())
        self.assertEqual(sync.stats["uncertain_clock_frames"], 1)
        self.assertEqual(sync.stats["accepted_frames"], 0)
        # A measured clock calibration update must also update fusion's camera
        # configuration before processing the newly accepted source packets.
        sync.connect_camera("a", clock_uncertainty_seconds=.005)
        fusion = MultiCameraFusion([camera("a", uncertainty=.005)], max_span_seconds=.025)
        sync.add_frame(packet("a", captured=10.1))
        bundle, = sync.flush()
        result = fusion.fuse(bundle)
        self.assertEqual(result.missing_cameras, ())
        self.assertEqual(result.persons, ())
        self.assertEqual(sync.stats["accepted_frames"], 1)

    def test_single_camera_uncertainty_exact_boundary_reaches_real_fusion(self):
        cameras = [camera("a", uncertainty=.0125)]
        sync = self.sync(cameras, max_skew_seconds=.025)
        sync.add_frame(packet("a"))
        bundle, = sync.flush()
        result = MultiCameraFusion(cameras, max_span_seconds=.025).fuse(bundle)
        self.assertEqual(result.bundle_id, bundle.bundle_id)
        self.assertEqual(sync.stats["uncertain_clock_frames"], 0)

    def test_pairwise_uncertainty_cannot_be_hidden_by_three_camera_chain(self):
        sync = self.sync([camera("a", uncertainty=.003), camera("b", uncertainty=.003), camera("c", uncertainty=.003)], max_skew_seconds=.025)
        for name, timestamp in (("a", 10), ("b", 10.012), ("c", 10.024)):
            sync.add_frame(packet(name, captured=timestamp))
        bundles = sync.flush()
        self.assertEqual([len(bundle.frames) for bundle in bundles], [2, 1])

    def test_full_bundle_span_not_adjacent_camera_gaps_is_bounded(self):
        sync = self.sync([camera(name) for name in "abc"], max_skew_seconds=.025)
        for name, timestamp in zip("abc", (10, 10.02, 10.04)):
            sync.add_frame(packet(name, captured=timestamp))
        self.assertEqual([len(bundle.frames) for bundle in sync.flush()], [2, 1])

    def test_skew_boundary_is_inclusive(self):
        sync = self.sync(max_skew_seconds=.025)
        sync.add_frame(packet("a", captured=10))
        sync.add_frame(packet("b", captured=10.025))
        self.assertEqual(len(sync.flush()[0].frames), 2)

    def test_unequal_camera_rates_never_reuse_slow_camera_frame(self):
        sync = self.sync(reorder_horizon_seconds=1)
        for sequence in range(4):
            sync.add_frame(packet("a", sequence, 10 + sequence * .01))
        sync.add_frame(packet("b", 0, 10.015))
        bundles = sync.flush()
        all_frames = [(frame.camera_id, frame.sequence) for bundle in bundles for frame in bundle.frames]
        self.assertEqual(len(all_frames), 5)
        self.assertEqual(len(set(all_frames)), 5)
        self.assertEqual(sum(frame.camera_id == "b" for bundle in bundles for frame in bundle.frames), 1)

    def test_disconnected_camera_packets_are_dropped_and_buffers_are_cleared(self):
        sync = self.sync()
        sync.add_frame(packet("b"))
        sync.disconnect_camera("b")
        sync.add_frame(packet("b", 1, 10.1))
        sync.add_frame(packet("a", 0, 10.1))
        bundle, = sync.flush()
        self.assertEqual(bundle.missing_cameras, ("b",))
        self.assertEqual(sync.stats["inactive_frames"], 1)
        self.assertEqual(sync.stats["disconnected_buffer_frames"], 1)

    def test_implicit_stream_epoch_reset_is_rejected(self):
        sync = self.sync()
        sync.add_frame(packet("a", 4))
        sync.add_frame(packet("a", 0, 10.1, epoch="restart"))
        self.assertEqual(sync.stats["epoch_mismatch_frames"], 1)

    def test_capture_clock_reset_requires_explicit_epoch_even_inside_reorder_horizon(self):
        sync = self.sync(reorder_horizon_seconds=1)
        sync.add_frame(packet("a", 20, 10.2))
        sync.add_frame(packet("a", 21, 10.1))
        self.assertEqual(sync.stats["nonmonotonic_capture_frames"], 1)
        self.assertEqual(sync.stats["accepted_frames"], 1)

    def test_reordered_sequence_with_inverted_capture_clock_is_rejected(self):
        sync = self.sync(reorder_horizon_seconds=1)
        sync.add_frame(packet("a", 21, 10.1))
        sync.add_frame(packet("a", 20, 10.2))
        self.assertEqual(sync.stats["nonmonotonic_capture_frames"], 1)

    def test_explicit_reconnect_resets_sequence_and_rejects_delayed_old_epoch(self):
        sync = self.sync()
        sync.add_frame(packet("a", 99, 10))
        sync.disconnect_camera("a")
        sync.connect_camera("a", stream_epoch="restart")
        sync.add_frame(packet("a", 0, 10.1, epoch="restart"))
        sync.add_frame(packet("a", 100, 10.2, epoch="default"))
        bundle, = sync.flush()
        self.assertEqual(bundle.frames[0].sequence, 0)
        self.assertEqual(bundle.frames[0].stream_epoch, "restart")
        self.assertEqual(sync.stats["epoch_mismatch_frames"], 1)
        with self.assertRaises(SynchronizationError):
            sync.connect_camera("a", stream_epoch="default")

    def test_resume_same_stream_keeps_replay_history(self):
        sync = self.sync()
        frame = packet("a", 99, 10)
        sync.add_frame(frame)
        sync.disconnect_camera("a")
        sync.connect_camera("a")
        sync.add_frame(frame)
        self.assertEqual(sync.stats["duplicate_frames"], 1)

    def test_clock_update_discards_pending_old_interpretation(self):
        sync = self.sync()
        sync.add_frame(packet("a", 0, 10))
        sync.connect_camera("a", stream_epoch="clock-v2", clock_offset_seconds=-1)
        sync.add_frame(packet("a", 0, 11.1, received=10.2, epoch="clock-v2"))
        sync.add_frame(packet("b", 0, 10.1))
        bundle, = sync.flush()
        self.assertEqual(len(bundle.frames), 2)
        self.assertAlmostEqual(bundle.target_timestamp_seconds, 10.1)
        self.assertEqual(sync.stats["disconnected_buffer_frames"], 1)

    def test_flush_seals_finished_interval_but_allows_new_capture(self):
        sync = self.sync()
        sync.add_frame(packet("a", 0, 10))
        sync.add_frame(packet("b", 0, 10.02))
        sync.flush()
        sync.add_frame(packet("a", 1, 10.01))
        sync.add_frame(packet("a", 2, 10.03))
        self.assertEqual(sync.stats["late_frames"], 1)
        self.assertEqual(len(sync.flush()), 1)

    def test_packet_and_replay_memory_remain_bounded_under_long_reordering(self):
        sync = self.sync([camera("a")], reorder_horizon_seconds=100, max_buffered_frames_per_camera=5, replay_window_size=8)
        for sequence in range(100):
            sync.add_frame(packet("a", sequence, 10 + sequence * .01))
        self.assertEqual(sync.stats["buffered_frames"], 5)
        self.assertEqual(sync.stats["replay_entries"], 8)
        self.assertEqual(sync.stats["dropped_buffer_frames"], 95)
        sync.add_frame(packet("a", 0, 10))
        self.assertEqual(sync.stats["replay_window_frames"], 1)
        self.assertEqual(sum(len(bundle.frames) for bundle in sync.flush()), 5)

    def test_epoch_tombstone_memory_is_bounded(self):
        sync = self.sync(max_retired_stream_epochs=2)
        sync.add_frame(packet("a"))
        sync.connect_camera("a", stream_epoch="one")
        sync.connect_camera("a", stream_epoch="two")
        with self.assertRaises(SynchronizationError):
            sync.connect_camera("a", stream_epoch="three")
        self.assertEqual(sync.stats["retired_epochs"], 2)

    def test_unknown_camera_has_no_effect_and_malformed_object_raises(self):
        sync = self.sync()
        sync.add_frame(packet("unknown"))
        self.assertEqual(sync.stats["unknown_camera_frames"], 1)
        with self.assertRaises(SynchronizationError):
            sync.add_frame(object())

    def test_stats_snapshot_is_independent(self):
        sync = self.sync()
        snapshot = sync.stats
        snapshot["accepted_frames"] = 500
        self.assertEqual(sync.stats["accepted_frames"], 0)

    def test_default_session_ids_do_not_collide_on_process_reconstruction(self):
        first = FrameSynchronizer([camera("a")])
        second = FrameSynchronizer([camera("a")])
        first.add_frame(packet("a"))
        second.add_frame(packet("a"))
        self.assertNotEqual(first.flush()[0].bundle_id, second.flush()[0].bundle_id)

    def test_seeded_four_camera_network_jitter_recovers_all_800_frames(self):
        rng = random.Random(839)
        cameras = [camera("a", .4), camera("b", -.7), camera("c", .02), camera("d", -.01)]
        sync = self.sync(cameras, max_skew_seconds=.010, reorder_horizon_seconds=.150)
        packets = []
        for sequence in range(200):
            for config in cameras:
                capture = 20 + sequence / 30 + rng.uniform(-.002, .002)
                packets.append(packet(config.camera_id, sequence, capture - config.clock_offset_seconds,
                                      capture + rng.uniform(.002, .080)))
        packets.sort(key=lambda frame: frame.received_at_seconds)
        bundles = []
        for frame in packets:
            bundles.extend(sync.add_frame(frame))
        bundles.extend(sync.flush())
        self.assertEqual(len(bundles), 200)
        self.assertTrue(all(len(bundle.frames) == 4 for bundle in bundles))
        self.assertTrue(all(len({frame.sequence for frame in bundle.frames}) == 1 for bundle in bundles))
        self.assertEqual(sync.stats["accepted_frames"], 800)
        self.assertEqual(sync.stats["emitted_frames"], 800)
        self.assertEqual(sync.stats["late_frames"], 0)
        self.assertLessEqual(max(bundle.span_seconds for bundle in bundles), .010)

    def test_concurrent_camera_ingestion_keeps_each_frame_exactly_once(self):
        cameras = [camera(name) for name in "abcd"]
        sync = self.sync(cameras, reorder_horizon_seconds=100, max_buffered_frames_per_camera=200)
        def ingest(config):
            for sequence in range(100):
                sync.add_frame(packet(config.camera_id, sequence, 20 + sequence * .033))
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(ingest, cameras))
        bundles = sync.flush()
        identifiers = [(frame.camera_id, frame.stream_epoch, frame.sequence) for bundle in bundles for frame in bundle.frames]
        self.assertEqual(len(bundles), 100)
        self.assertEqual(len(identifiers), 400)
        self.assertEqual(len(set(identifiers)), 400)
        self.assertEqual(sync.stats["accepted_frames"], 400)

    def test_invalid_configuration_is_rejected(self):
        for kwargs in (
            {"max_skew_seconds": -1}, {"max_skew_seconds": float("nan")},
            {"max_skew_seconds": True}, {"reorder_horizon_seconds": float("inf")},
            {"max_buffered_frames_per_camera": 0}, {"max_buffered_frames_per_camera": 1.5},
            {"max_frame_age_seconds": 0}, {"max_future_lead_seconds": -.1},
            {"replay_window_size": True}, {"max_retired_stream_epochs": 0}, {"session_id": ""},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.sync(**kwargs)
        with self.assertRaises(ValueError):
            FrameSynchronizer([])
        with self.assertRaises(ValueError):
            FrameSynchronizer([camera("a"), camera("a")])

    def test_invalid_lifecycle_inputs_leave_stream_usable(self):
        sync = self.sync()
        for kwargs in ({"stream_epoch": "\n"}, {"clock_offset_seconds": float("inf")},
                       {"clock_offset_seconds": True}, {"clock_uncertainty_seconds": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                sync.connect_camera("a", **kwargs)
        for now in (-1, float("nan"), float("inf"), True):
            with self.subTest(now=now), self.assertRaises(ValueError):
                sync.advance_watermark(now)
        for operation in (sync.connect_camera, sync.disconnect_camera):
            with self.assertRaises(ValueError):
                operation("absent")
        sync.add_frame(packet("a"))
        self.assertEqual(len(sync.flush()), 1)


if __name__ == "__main__":
    unittest.main()
