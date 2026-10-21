import unittest

from ui.worker_state import FaceRecognitionState, RegistrationCaptureState, StablePrimaryLabelTracker


class FaceRecognitionStateTest(unittest.TestCase):
    def setUp(self):
        self.state = FaceRecognitionState(
            unknown_threshold_seconds=1.0, known_threshold_seconds=1.5,
            switch_penalty_seconds=0.6, min_observation_seconds=0.05,
            label_tracker=StablePrimaryLabelTracker(60, 20, 6, 4, 1),
        )

    def test_new_person_cannot_inherit_previous_person_confirmation(self):
        self.assertFalse(self.state.observe_recognized("Alice", 1.0))
        self.assertTrue(self.state.observe_recognized("Alice", 0.5))
        self.assertFalse(self.state.observe_recognized("Bob", 0.1))
        self.assertAlmostEqual(self.state.known_elapsed_seconds, 0.1)
        self.assertFalse(self.state.observe_recognized("Bob", 0.5))

    def test_unknown_or_missing_face_requires_fresh_confirmation(self):
        for interrupt in (self.state.observe_unknown, self.state.observe_no_face):
            with self.subTest(interrupt=interrupt.__name__):
                self.state.reset()
                self.state.observe_recognized("Alice", 1.4)
                interrupt(0.05)
                self.assertFalse(self.state.observe_recognized("Alice", 0.1))
                self.assertAlmostEqual(self.state.known_elapsed_seconds, 0.1)

    def test_invalid_elapsed_time_does_not_create_recognition_evidence(self):
        for duration in (float("nan"), float("inf"), -1, "invalid", None, 0):
            self.state.reset()
            self.assertFalse(self.state.observe_recognized("Alice", duration))
            self.assertEqual(self.state.known_elapsed_seconds, 0)

    def test_registration_cannot_advance_without_an_encoding(self):
        capture = RegistrationCaptureState(1, 1, 2)
        capture.start("Alice")
        capture.add_valid_encoding(None, 1)
        self.assertEqual(capture.valid_capture_seconds, 0)
        self.assertFalse(capture.is_complete)
        capture.add_valid_encoding([0.2] * 128, 1)
        self.assertTrue(capture.is_complete)
