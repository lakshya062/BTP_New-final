import concurrent.futures
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from core import face_run


class FaceModelTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = str(Path(self.directory.name) / "faces.hdf5")
        self.encodings = np.full((2, 128), 0.2, dtype=np.float32)

    def test_unicode_identity_round_trip_and_delete(self):
        self.assertTrue(face_run.append_user_to_model_hdf5("लक्ष्य", self.encodings, self.path))
        encodings, names = face_run.load_trained_model_hdf5(self.path)
        self.assertEqual(names, ["लक्ष्य", "लक्ष्य"])
        np.testing.assert_array_equal(encodings, self.encodings)
        if os.name != "nt":
            self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        self.assertTrue(face_run.delete_user_from_model_hdf5("लक्ष्य", self.path))
        self.assertEqual(face_run.load_trained_model_hdf5(self.path), ([], []))

    def test_invalid_names_and_embeddings_never_change_existing_model(self):
        self.assertTrue(face_run.append_user_to_model_hdf5("Alice", self.encodings, self.path))
        original = Path(self.path).read_bytes()
        for name, encodings in [
            ("nested/name", self.encodings), ("", self.encodings),
            ("Unknown", self.encodings), ("Bob", np.zeros((2, 127))),
            ("Bob", np.full((1, 128), np.nan)), ("Bob", []),
        ]:
            with self.subTest(name=name, shape=np.shape(encodings)):
                self.assertFalse(face_run.append_user_to_model_hdf5(name, encodings, self.path))
                self.assertEqual(Path(self.path).read_bytes(), original)

    def test_failed_publish_preserves_model_and_cleans_temporary_file(self):
        self.assertTrue(face_run.append_user_to_model_hdf5("Alice", self.encodings, self.path))
        original = Path(self.path).read_bytes()
        with patch.object(face_run.os, "replace", side_effect=OSError("disk failure")):
            self.assertFalse(face_run.append_user_to_model_hdf5("Bob", self.encodings, self.path))
            self.assertFalse(face_run.delete_user_from_model_hdf5("Alice", self.path))
        self.assertEqual(Path(self.path).read_bytes(), original)
        self.assertEqual(list(Path(self.directory.name).iterdir()), [Path(self.path)])

    def test_legacy_migration_is_private_and_failed_copy_leaves_no_partial_model(self):
        self.assertTrue(face_run.append_user_to_model_hdf5("Alice", self.encodings, self.path))
        destination = str(Path(self.directory.name) / "runtime.hdf5")
        with patch.dict(os.environ, {face_run.FACE_MODEL_PATH_ENV: ""}), patch.object(
            face_run, "project_path", return_value=self.path
        ), patch.object(face_run, "state_path", return_value=destination):
            with patch.object(face_run.os, "replace", side_effect=OSError("disk failure")):
                self.assertEqual(face_run.resolve_face_model_path(), destination)
            self.assertFalse(Path(destination).exists())
            self.assertEqual(face_run.resolve_face_model_path(), destination)
        self.assertEqual(face_run.load_trained_model_hdf5(destination)[1], ["Alice", "Alice"])
        if os.name != "nt":
            self.assertEqual(os.stat(destination).st_mode & 0o777, 0o600)

    def test_repeated_registration_bounds_samples_on_disk(self):
        samples = np.full((100, 128), 0.2, dtype=np.float32)
        for _ in range(3):
            self.assertTrue(face_run.append_user_to_model_hdf5("Alice", samples, self.path))
        with h5py.File(self.path, "r") as model:
            self.assertEqual(model["Alice/encodings"].shape, (face_run.MAX_ENCODINGS_PER_USER, 128))
            self.assertEqual(model["Alice/names"].shape, (face_run.MAX_ENCODINGS_PER_USER,))

    def test_corrupt_group_cannot_misalign_other_identities(self):
        with h5py.File(self.path, "w") as model:
            model.create_dataset("Broken/encodings", data=self.encodings)
            model.create_dataset("Broken/names", data=[b"Wrong"])
            model.create_dataset("Invalid/encodings", data=np.full((1, 128), np.inf))
            model.create_dataset("Invalid/names", data=[b"Invalid"])
            model.create_dataset("Valid/encodings", data=self.encodings)
            model.create_dataset("Valid/names", data=[b"Valid", b"Valid"])
        encodings, names = face_run.load_trained_model_hdf5(self.path)
        self.assertEqual(names, ["Valid", "Valid"])
        self.assertEqual(len(encodings), len(names))

    def test_concurrent_recognizers_do_not_lose_users(self):
        first = face_run.FaceRecognizer(self.path)
        second = face_run.FaceRecognizer(self.path)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            registrations = [
                executor.submit(first.register_new_user, "Alice", self.encodings),
                executor.submit(second.register_new_user, "Bob", self.encodings),
            ]
            self.assertTrue(all(job.result() for job in registrations))
        _, names = face_run.load_trained_model_hdf5(self.path)
        self.assertEqual(set(names), {"Alice", "Bob"})

    def test_identity_rejects_ambiguous_and_nonfinite_distances(self):
        indices = face_run._build_name_to_indices(["Alice", "Bob"])
        self.assertEqual(face_run._resolve_identity(np.array([0.2, 0.21]), indices, 0.45, 0.03), "Unknown")
        self.assertEqual(face_run._resolve_identity(np.array([0.2, np.nan]), indices, 0.45, 0.03), "Unknown")
        self.assertEqual(face_run._resolve_identity(np.array([0.2, 0.4]), indices, 0.45, 0.03), "Alice")

    def test_invalid_gallery_cannot_assign_face_to_wrong_label(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch.object(face_run.face_recognition, "face_locations", return_value=[(0, 20, 20, 0)]), patch.object(
            face_run.face_recognition, "face_encodings", return_value=[self.encodings[0]]
        ):
            _, locations, names = face_run.process_frame((frame, self.encodings, ["Alice"]))
        self.assertEqual(len(locations), 1)
        self.assertEqual(names, ["Unknown"])

    def test_invalid_inference_config_does_not_escape_frame_boundary(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result, locations, names = face_run.process_frame((frame, [], [], {"frame_scale": "bad"}))
        self.assertIs(result, frame)
        self.assertEqual((locations, names), ([], []))
